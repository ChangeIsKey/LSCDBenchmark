# Setting Docstrings

To auto-document your code, you need to follow the docstring format in your code. For example:

```python
def predict(self, lemma: Lemma) -> float:
    """Generates predictions for use pair samples for input lemma.

    :param lemma: lemma instance from data set
    :type lemma: Lemma
    :return: mean of pairwise distances
    :rtype: float
    """              
    use_pairs = lemma.use_pairs(
        group=self.use_pair_options.group, 
        sample=self.use_pair_options.sample
    )
    similarities = self.wic.predict(use_pairs)
    return -np.mean(similarities).item()
```

```{note}
If you use VS Code as the source-code editor, the extenstion [autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) is highly recommended.
```
