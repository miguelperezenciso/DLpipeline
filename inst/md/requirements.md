### About requirements 

Our pipeline sucefully run under the following version of the dependencies:


talos==0.4.8

seaborn==0.9.0

wrangle==0.6.7

scipy==1.2.0

matplotlib==2.2.3

numpy==1.16.4

pandas==0.24.2

Keras==2.2.4

scikit_learn==0.21.3 

Note that talos versions higher than 0.4.8 have some modifications of key functions, e.g. `fmeasure_acc` and `talos.model.layers` have been changed in the new talos version. The  `live` in callbacks does not works for newest talos version either. Please make sure you have the right versions before run the examples. 

You can install these versions by typing in bash:

`python3 -m pip install -r requirements.txt` 

probably as sudoer. 


### Can you use the newest talos version? 

Of course, you  still could use a newest version, but to succefully run the examples, you have change some key commands, e.g.  

Please replace 

```python
import talos as ta
import wrangle as wr
from talos.metrics.keras_metrics import fmeasure_acc
from talos.model.layers import hidden_layers
from talos import live
from talos.model import lr_normalizer, early_stopper, hidden_layers
```

by 

```python
import keras
import talos as ta
import wrangle as wr
from talos.utils import hidden_layers
from talos.model import lr_normalizer, early_stopper
```
`f_measure_acc` is just `acc` from keras. Then, you have to replace  `f_measure_acc` by `acc` in the code since you already imported keras. Talos also have other metrics as options, please check https://github.com/autonomio/talos/blob/master/docs/Metrics.md. The hidden_layers command is now in talos.utils, whereas live is not requested. 

The PDL_talos_3.6.ipynb  contains the example that run with talos==0.3.6


