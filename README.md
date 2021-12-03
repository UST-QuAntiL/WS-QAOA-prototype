# WS-QAOA-prototype

This repository contains the prototypical implementation used to generate results presented in our publication **"Selection and Optimization of Hyperparameters in Warm-started Quantum Optimization for the MaxCut Problem"** [1]

In [WsQaoaRunner.py](WsQaoaRunner.py) we defined the high level function *run_ws_qaoa* to execute the WS-QAOA algorithm with different hyperparameter settings. 
The evaluation is done using *execute_circuit* in combination with an objective function (defined in [objectiveFunctions.py](objectiveFunctions.py)). 
Besides the actual objective functions compared in the work, there are two for evaluation: *F_MaxCut_probability* and *F_BetterCut_probability*.

The three optimization strategies are implemented in [optimization/OptimizationStrategies.py](optimization/OptimizationStrategies.py).

The problem instances are stored in [graphs](graphs). [Set 1](graphs/set-1) was used for most of the experiments, wheras [Set 2](graphs/set-2) was used additionally in experiment 2 to confirm the performance of the optimization strategies on a larger set of graphs.

The individual experiments are sketched in [experiment1.py](experiment1.py), [experiment2.py](experiment2.py), [experiment3.1.py](experiment3.1.py), and [experiment3.2.py](experiment3.2.py).

In [results](results), the generated results can be found, including some additional results that were not presented in the paper for brevity.

### References
[1] todo: link to arxiv

### Disclaimer of Warranty

Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.

### Haftungsausschluss

Dies ist ein Forschungsprototyp. Die Haftung für entgangenen Gewinn, Produktionsausfall, Betriebsunterbrechung, entgangene Nutzungen, Verlust von Daten und Informationen, Finanzierungsaufwendungen sowie sonstige Vermögens- und Folgeschäden ist, außer in Fällen von grober Fahrlässigkeit, Vorsatz und Personenschäden, ausgeschlossen.
