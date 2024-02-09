This folder contains synthetic data we generated using the OpenAI API.
The general idea was to permute the context of a citation in such a way that 
it is opinionated. We then wanted to use this synthetic data to fine-tune
a XL-Net, but the synthetic data was too much opinionated and also
the Few-Shot-Chain-of-Thought wasn't accurate enough to use it for labeling.
SO due to that we omitted the idea of fine-tuning a own model and marked this
idea as failed.