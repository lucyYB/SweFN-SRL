# SweFN-SRL
End-to-end semantic role labeller (SRL) based on Swedish FrameNet (SweFN)

This Swedish SRL model can, with a given Swedish sentence, perform 
  - trigger identification, 
  - frame classification, 
  - argument extraction 
 tasks automatically in a series. 
 
Model 1:

This Swedish SRL is based on a pre-trained English SRL (T5 based, https://github.com/chanind/frame-semantic-transformer.git).
There are two versions: "small", and "base", which are corresponding to the two versions of the English SRL model.

Model 2:

This Swedish SRL is based on the multilingual T5 (mT5), only small version is fine-tuned.
