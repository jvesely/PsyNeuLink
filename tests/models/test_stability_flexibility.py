import psyneulink as pnl
import numpy as np
import pytest


# Define function to generate a counterbalanced trial sequence with a specified switch trial frequency
def generateTrialSequence(N, Frequency):

    # Compute trial number
    nTotalTrials = N
    switchFrequency = Frequency

    nSwitchTrials = int(nTotalTrials * switchFrequency)
    nRepeatTrials = int(nTotalTrials - nSwitchTrials)

    # Determine task transitions
    transitions = [1] * nSwitchTrials + [0] * nRepeatTrials
    order = np.random.permutation(list(range(nTotalTrials)))
    transitions[:] = [transitions[i] for i in order]

    # Determine stimuli with 50% congruent trials
    stimuli = [[1, 1]] * int(nSwitchTrials / 4) + [[1, -1]] * int(nSwitchTrials / 4) + [[-1, -1]] * int(nSwitchTrials / 4) + [[-1, 1]] * int(nSwitchTrials / 4) + \
              [[1, 1]] * int(nRepeatTrials / 4) + [[1, -1]] * int(nRepeatTrials / 4) + [[-1, -1]] * int(nRepeatTrials / 4) + [[-1, 1]] * int(nRepeatTrials / 4)
    stimuli[:] = [stimuli[i] for i in order]

    #stimuli[:] = [[1, 1]] * nTotalTrials

    # Determine cue-stimulus intervals
    CSI = [1200] * int(nSwitchTrials / 8) + [1200] * int(nSwitchTrials / 8) + \
          [1200] * int(nSwitchTrials / 8) + [1200] * int(nSwitchTrials / 8) + \
          [1200] * int(nSwitchTrials / 8) + [1200] * int(nSwitchTrials / 8) + \
          [1200] * int(nSwitchTrials / 8) + [1200] * int(nSwitchTrials / 8) + \
          [1200] * int(nRepeatTrials / 8) + [1200] * int(nRepeatTrials / 8) + \
          [1200] * int(nRepeatTrials / 8) + [1200] * int(nRepeatTrials / 8) + \
          [1200] * int(nRepeatTrials / 8) + [1200] * int(nRepeatTrials / 8) + \
          [1200] * int(nRepeatTrials / 8) + [1200] * int(nRepeatTrials / 8)
    CSI[:] = [CSI[i] for i in order]

    # Set the task order
    tasks = [[1, 0]] * (nTotalTrials + 1)
    for i in list(range(nTotalTrials)):
        if transitions[i] == 0:
            tasks[i + 1] = tasks[i]
        if transitions[i] == 1:
            if tasks[i] == [1, 0]:
                tasks[i + 1] = [0, 1]
            if tasks[i] == [0, 1]:
                tasks[i + 1] = [1, 0]
    tasks = tasks[1:]

    #Determine correct response based on stimulus and task input
    correctResponse = np.sum(np.multiply(tasks, stimuli), axis=1)

    # # Check whether combinations of transitions, stimuli and CSIs are counterbalanced

    # # This is used later to check whether trials are counterbalanced
    # stimuli_type = [1] * int(nSwitchTrials/4) + [2] * int(nSwitchTrials/4) + [3] * int(nSwitchTrials/4) + [4] * int(nSwitchTrials/4) + \
    #           [1] * int(nRepeatTrials/4) + [2] * int(nRepeatTrials/4) + [3] * int(nRepeatTrials/4) + [4] * int(nRepeatTrials/4)
    # stimuli_type[:] = [stimuli_type[i] for i in order]

    # Trials = pd.DataFrame({'TrialType': transitions,
    #                        'Stimuli': stimuli_type,
    #                        'CSI': CSI
    #                        }, columns= ['TrialType', 'Stimuli', 'CSI'])
    #
    # trial_counts = Trials.pivot_table(index=['TrialType', 'Stimuli', 'CSI'], aggfunc='size')
    # print (trial_counts)

    return tasks, stimuli, CSI, correctResponse


# Stability-Flexibility Model
#@pytest.mark.stress
@pytest.mark.parametrize("num_generators", [3,
                                            pytest.param(100000, marks=pytest.mark.stress)])
@pytest.mark.parametrize("mode", pytest.helpers.get_comp_execution_modes() +
                                 [pytest.helpers.cuda_param('Python-PTX'),
                                  pytest.param('Python-LLVM', marks=pytest.mark.llvm)])
@pytest.mark.parametrize('prng', ['Default', 'Philox'])
@pytest.mark.parametrize('fp_type', [pnl.core.llvm.ir.DoubleType, pnl.core.llvm.ir.FloatType])
@pytest.mark.benchmark
def test_stability_flexibility(mode, benchmark, num_generators, prng, fp_type):
    if mode == pnl.ExecutionMode.LLVM:
        pytest.skip("takes too long to compile")

    if str(mode).startswith('Python-'):
        ocm_mode = mode.split('-')[1]
        mode = pnl.ExecutionMode.Python
    else:
        ocm_mode = 'Python'

    pnl.core.llvm.builder_context.LLVMBuilderContext.default_float_ty = fp_type()

    # 16 is minimum number of trial inputs that can be generated
    taskTrain, stimulusTrain, cueTrain, correctResponse = generateTrialSequence(16, 0.5)

    GAIN = 1.0
    LEAK = 1.0
    COMP = 7.5
    AUTOMATICITY = 0.15 # Automaticity Weight

    STARTING_POINT = 0.0
    THRESHOLD = 0.2
    NOISE = 0.1
    SCALE = 1 # Scales DDM inputs so threshold can be set to 1

    # Task Layer: [Color, Motion] {0, 1} Mutually Exclusive
    # Origin Node
    taskLayer = pnl.TransferMechanism(size=2,
                                      function=pnl.Linear(slope=1, intercept=0),
                                      output_ports=[pnl.RESULT],
                                      name='Task Input [I1, I2]')

    # Stimulus Layer: [Color Stimulus, Motion Stimulus]
    # Origin Node
    stimulusInfo = pnl.TransferMechanism(size=2,
                                         function=pnl.Linear(slope=1, intercept=0),
                                         output_ports=[pnl.RESULT],
                                         name="Stimulus Input [S1, S2]")

    # Cue-To-Stimulus Interval Layer
    # Origin Node
    cueInterval = pnl.TransferMechanism(size=1,
                                        function=pnl.Linear(slope=1, intercept=0),
                                        output_ports=[pnl.RESULT],
                                        name='Cue-Stimulus Interval')

    # Correct Response Info
    # Origin Node
    correctResponseInfo = pnl.TransferMechanism(size=1,
                                                function=pnl.Linear(slope=1, intercept=0),
                                                output_ports=[pnl.RESULT],
                                                name='Correct Response Info')

    # Control Module Layer: [Color Activation, Motion Activation]
    controlModule = pnl.LCAMechanism(size=2,
                                     function=pnl.Logistic(gain=GAIN),
                                     leak=LEAK,
                                     competition=COMP,
                                     self_excitation=0,
                                     noise=0,
                                     termination_measure=pnl.TimeScale.TRIAL,
                                     termination_threshold=1200,
                                     time_step_size=0.1,
                                     name='Task Activations [Act1, Act2]')

    # Control Mechanism Setting Cue-To-Stimulus Interval
    csiController = pnl.ControlMechanism(monitor_for_control=cueInterval,
                                         control_signals=pnl.ControlSignal(
                                            modulates=(pnl.TERMINATION_THRESHOLD, controlModule),
                                            default_allocation=1200
                                         ),
                                         modulation=pnl.OVERRIDE)

    # Hadamard product of controlModule and Stimulus Information
    nonAutomaticComponent = pnl.TransferMechanism(size=2,
                                                  function=pnl.Linear(slope=1, intercept=0),
                                                  input_ports=pnl.InputPort(combine=pnl.PRODUCT),
                                                  output_ports=[pnl.RESULT],
                                                  name='Non-Automatic Component [S1*Act1, S2*Act2]')

    # Multiply Stimulus Input by the automaticity weight
    congruenceWeighting = pnl.TransferMechanism(size=2,
                                                function=pnl.Linear(slope=AUTOMATICITY, intercept=0),
                                                output_ports=[pnl.RESULT],
                                                name="Automaticity-weighted Stimulus Input [w*S1, w*S2]")

    # Summation of nonAutomatic and Automatic Components
    ddmCombination = pnl.TransferMechanism(size=1,
                                           function=pnl.Linear(slope=1, intercept=0),
                                           input_ports=pnl.InputPort(combine=pnl.SUM),
                                           output_ports=[pnl.RESULT],
                                           name="Drift = (w*S1 + w*S2) + (S1*Act1 + S2*Act2)")

    # Ensure upper boundary of DDM is always correct response by multiplying DDM input by correctResponseInfo
    ddmRecodeDrift = pnl.TransferMechanism(size=1,
                                          function=pnl.Linear(slope=1, intercept=0),
                                          input_ports=pnl.InputPort(combine=pnl.PRODUCT),
                                          output_ports=[pnl.RESULT],
                                          name='Recoded Drift = Drift * correctResponseInfo')

    # Scale DDM inputs
    ddmInputScale = pnl.TransferMechanism(size=1,
                                          function=pnl.Linear(slope=SCALE, intercept=0),
                                          output_ports=[pnl.RESULT],
                                          name='Scaled DDM Input')

    # Decision Module
    decisionMaker = pnl.DDM(function=pnl.DriftDiffusionIntegrator(non_decision_time=STARTING_POINT,
                                                                  threshold=THRESHOLD,
                                                                  noise=np.sqrt(NOISE),
                                                                  time_step_size=0.001),
                            output_ports=[pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME],
                            reset_stateful_function_when=pnl.Never(),
                            execute_until_finished=False,
                            max_executions_before_finished=1000,
                            name='DDM')

    # Composition Creation
    stabilityFlexibility = pnl.Composition(controller_mode=pnl.BEFORE)

    # Node Creation
    stabilityFlexibility.add_node(taskLayer)
    stabilityFlexibility.add_node(stimulusInfo)
    stabilityFlexibility.add_node(cueInterval)
    stabilityFlexibility.add_node(correctResponseInfo)
    stabilityFlexibility.add_node(controlModule)
    stabilityFlexibility.add_node(csiController)
    stabilityFlexibility.add_node(nonAutomaticComponent)
    stabilityFlexibility.add_node(congruenceWeighting)
    stabilityFlexibility.add_node(ddmCombination)
    stabilityFlexibility.add_node(ddmRecodeDrift)
    stabilityFlexibility.add_node(ddmInputScale)
    stabilityFlexibility.add_node(decisionMaker)

    # Projection Creation
    stabilityFlexibility.add_projection(sender=taskLayer, receiver=controlModule)
    stabilityFlexibility.add_projection(sender=controlModule, receiver=nonAutomaticComponent)
    stabilityFlexibility.add_projection(sender=stimulusInfo, receiver=nonAutomaticComponent)
    stabilityFlexibility.add_projection(sender=stimulusInfo, receiver=congruenceWeighting)
    stabilityFlexibility.add_projection(sender=nonAutomaticComponent, receiver=ddmCombination)
    stabilityFlexibility.add_projection(sender=congruenceWeighting, receiver=ddmCombination)
    stabilityFlexibility.add_projection(sender=ddmCombination, receiver=ddmRecodeDrift)
    stabilityFlexibility.add_projection(sender=correctResponseInfo, receiver=ddmRecodeDrift)
    stabilityFlexibility.add_projection(sender=ddmRecodeDrift, receiver=ddmInputScale)
    stabilityFlexibility.add_projection(sender=ddmInputScale, receiver=decisionMaker)

    # Restrict non-mutable nodes to run only in the first pass
#    stabilityFlexibility.scheduler.add_condition(taskLayer, pnl.AtPass(0))
#    stabilityFlexibility.scheduler.add_condition(stimulusInfo, pnl.AtPass(0))
#    stabilityFlexibility.scheduler.add_condition(correctResponseInfo, pnl.AtPass(0))
#    stabilityFlexibility.scheduler.add_condition(congruenceWeighting, pnl.AtPass(0))

    # Hot-fix currently necessary to allow control module and DDM to execute in parallel in compiled mode
    # We need two gates in order to output both values (decision and response) from the ddm
    decisionGate = pnl.ProcessingMechanism(size=1, name="DECISION_GATE")
    stabilityFlexibility.add_node(decisionGate)

    responseGate = pnl.ProcessingMechanism(size=1, name="RESPONSE_GATE")
    stabilityFlexibility.add_node(responseGate)

    stabilityFlexibility.add_projection(sender=decisionMaker.output_ports[0], receiver=decisionGate)
    stabilityFlexibility.add_projection(sender=decisionMaker.output_ports[1], receiver=responseGate)


    # Add ObjectiveMechanism to store the values in 'saved_values'
    objectiveMech = pnl.ObjectiveMechanism(name="RESPONSE_Objective")
    #objectiveMech.function.parameters.weights = [0.5, 1.0]
    stabilityFlexibility.add_node(objectiveMech)
    stabilityFlexibility.add_projection(sender=responseGate, receiver=objectiveMech)
    stabilityFlexibility.add_projection(sender=decisionGate, receiver=objectiveMech)

    # Combine decision and response time
    # Response is either -0.2 or 0.2, multiply by 5 to get -1/1,
    # then multiply by the response time to get a distribution of
    # positive times for positive responses, and a distribution of
    # negative times for negative responses.
    assert len(objectiveMech.input_ports) == 1
    objectiveMech.input_port.function.weights = [[5.], [1.]]
    objectiveMech.input_port.function.operation = pnl.PRODUCT

    # Add controller to gather multiple samples
    stabilityFlexibility.add_controller(
        pnl.OptimizationControlMechanism(
            state_features=[taskLayer.input_port, stimulusInfo.input_port, cueInterval.input_port, correctResponseInfo.input_port],
            function=pnl.GridSearch(save_values=True),
            agent_rep=stabilityFlexibility,
            objective_mechanism=objectiveMech,
            comp_execution_mode=ocm_mode,
            control_signals=pnl.ControlSignal(
                modulates=('seed-function', decisionMaker),
                modulation=pnl.OVERRIDE,
                default_allocation=[num_generators],
                allocation_samples=pnl.SampleSpec(start=0, stop=num_generators - 1, step=1),
                cost_options=pnl.CostFunctions.NONE
            )
        )
    )
    if prng == 'Philox':
        stabilityFlexibility.controller.function.parameters.random_state.set(pnl.core.globals.utilities._SeededPhilox([0]))
        decisionMaker.parameters.random_state.set(pnl.core.globals.utilities._SeededPhilox([0]))
        decisionMaker.function.parameters.random_state.set(pnl.core.globals.utilities._SeededPhilox([0]))

    # Set scheduler conditions so that the gates are not executed (and hence the composition doesn't finish) until decisionMaker is finished
    stabilityFlexibility.scheduler.add_condition(decisionGate, pnl.WhenFinished(decisionMaker))
    stabilityFlexibility.scheduler.add_condition(responseGate, pnl.WhenFinished(decisionMaker))

    inputs = {taskLayer: taskTrain,
              stimulusInfo: stimulusTrain,
              cueInterval: cueTrain,
              correctResponseInfo: correctResponse}

    benchmark(stabilityFlexibility.run, inputs, execution_mode=mode, num_trials=1)

    if num_generators == 3:

        # saved values are only available in Python, and are overwritten after every invocation
        if mode == pnl.ExecutionMode.Python and not benchmark.enabled:
            is_fp32 = ocm_mode != 'Python' and fp_type.intrinsic_name.endswith('32')
            ocm_results = np.squeeze(stabilityFlexibility.controller.function.saved_values)

            if prng == 'Default' and is_fp32:
                np.testing.assert_allclose(ocm_results, [0.07699998, 0.26600012, 0.05999995])
            elif prng == 'Default':   # fp64 computation
                np.testing.assert_allclose(ocm_results, [0.077, 0.266, 0.06])
            elif prng == 'Philox' and is_fp32:
                np.testing.assert_allclose(ocm_results, [0.18600021, 0.10100003, 0.19800022])
            elif prng == 'Philox':   # fp64 computation
                np.testing.assert_allclose(ocm_results, [0.235, 0.086, 0.229])
            else:
                assert False, "Unknown PRNG and fp_type combination: {} {}".format(prng, str(fp_type))

        # The final ('max') results are different for the special case of mode == Python and
        # ocm_mode != Python. The OCM simulation is run in compiled mode using fp32, this
        # selects a "winning" set of parameters. Those parameters are then used to run the
        # composition in Python using fp64 precision.
        # This is particularly the case for Philox which produces different sequences for fp32
        # and fp64, even when using the same seed.
        is_fp32 = mode != pnl.ExecutionMode.Python and fp_type.intrinsic_name.endswith('32')
        if prng == 'Default' and is_fp32:
            np.testing.assert_allclose(stabilityFlexibility.results[0], [[1200.], [0.2], [0.26600012]])
        elif prng == 'Default':   # fp64 computation
            np.testing.assert_allclose(stabilityFlexibility.results[0], [[1200.], [0.2], [0.266]])
        elif prng == 'Philox' and is_fp32:
            np.testing.assert_allclose(stabilityFlexibility.results[0], [[1200.], [0.2], [0.19800022]])
        elif prng == 'Philox' and ocm_mode != 'Python' and fp_type.intrinsic_name.endswith('32'):
            # A special case of running fp32 simulations and fp64 final execution. It selects the
            # fp64 result at the same index (using the same seed) as the largest fp32 result (see above).
            np.testing.assert_allclose(stabilityFlexibility.results[0], [[1200.], [0.2], [0.229]])
        elif prng == 'Philox':   # fp64 computation
            np.testing.assert_allclose(stabilityFlexibility.results[0], [[1200.], [0.2], [0.235]])
        else:
            assert False, "Unknown PRNG and fp_type combination: {} - {}".format(prng,  fp_type.intrinsic_name)

    if num_generators == 100000 and mode == pnl.ExecutionMode.Python:
        data = stabilityFlexibility.controller.function.saved_values

        from matplotlib import pyplot as plt
        plt.hist(data, bins=600, range=[-3.0,3.0])
#        plt.show()
