# GANs
All new GAN should have a naming convention of 

`[generatorname].py`

With 3 functions of `import_all`, `train`, and `generate` that takes in parameters

```
import_model():
  import [whatever imports you want to add]
  # This is used to ensure that torch.load works
train(js, dataset, epochs=):
  import [whatever imports you want to add]
  ...
  return model
generate(js, model, sample_total=):
  import [whatever imports you want to add]
  ...
  return samples
```

`js` is a dictionary containing all the values defined in the `.json` file (such as `discrete_columns`) as well as runtime values (`file`, `folder`, `files`)

## Avoid imports outside of function
Adding imports outside of the `generate` function will slowdown the initial ramp up. Properly add `imports` at the beginning of the `generate` function.


## TODOS
add batchsize to CTGAN and tablegan