import pickle

class_names = ['Good query','Bad query']
lgs = pickle.load(open('./waf_model/pickled_lgs', 'rb'))
vectorizer = pickle.load(open('./waf_model/pickled_vectorizer','rb'))


# Explaining predictions using lime
# Lime explainers assume that classifiers act on raw text, but sklearn classifiers act on vectorized representation of texts. 
# For this purpose, we use sklearn's pipeline, and implements predict_proba on raw_text lists.
from lime import lime_text
from sklearn.pipeline import make_pipeline
prediction_pipeline = make_pipeline(vectorizer, lgs)



# Now we create an explainer object. We pass the class_names as an argument for prettier display.
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)


# We then generate an explanation
query = '<script>alert(1)/javascript/nets.png/javascript/nets.png/javascript/nets.png/javascript/nets.png/javascript/nets.png</script>'
# query = '/<script>alert(1)/*index.html/index.html/index.html/index.html/index.html/index.html/index.html/index.html/index.html/index.html/index.html/index.html/index.html/index.html/index.html/index.html/index.html/index.html/index.html/index.html/index.html/*/</script>/'
# query = '/moodle/filter/tex/texed.php?formdata=foo&pathname=foo"+||+echo+db+4d+5a+50+00+02+00+00+00+04+00+0f+00+ff+ff+00+00+b8+00+00+00+00+00+00+00+40++>>esbq'
exp = explainer.explain_instance(query, prediction_pipeline.predict_proba, num_features=6)
print('Probability =', prediction_pipeline.predict_proba([query])[0,1])


# The explanation is presented below as a list of weighted features. 
for feature, weight in exp.as_list():
    print(f"{feature}: {weight:.4f}")
# we can save the fully contained html page to a file:
output_path = '/tmp/waf_explanation.html'
exp.save_to_file(output_path)
print(f"\nLIME explanation saved as HTML file at: {output_path}")
