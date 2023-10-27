# Autoencoder for Product Matching

This was an experiment for a possible PhD topic. The main idea was to use different Autoencoder for entity resolution / product matching. The core idea was to pretrain an Autoencoder on the positive pairs (=matching entities) and use the output of the encoder to train a classifier. During pretraining the the task of the target of the Autoencoder would randomly change between reconstructing the data point and reconstructing the matching data point. That way the Autoencoder should learn to only focus on the common attributes of the positive paris. Here is an example training the Autoencoder:

```
# randomly change the input and the goal
ref_model_input = random.choice([input_a, input_b])
ref_expected = random.choice([input_a, input_b])

# clone them, so they won't reference the same object in memory
model_input = ref_model_input.detach().clone()
expected = ref_expected.detach().clone()

model_input, expected = self.move_to_device(model_input, expected)
expected = self.prepare_input(expected)

reconstructed = self.model(model_input)
loss = self.criterion(reconstructed, expected)
```

The idea got scrapped because Autoencoders in the entity resolution / product matching domain were not really new and the results of the code here were mediocre.


# Learnings

* Luigi framwork is good
* Mlflow tracking is very good for tracking experiments
* The project structure is fine

