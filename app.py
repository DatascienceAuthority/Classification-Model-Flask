from flask import Flask, render_template
import pickle
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField, validators, StringField, IntegerField, FloatField, SelectField
from wtforms.fields.html5 import EmailField
from flask_wtf.file import FileField, FileRequired, FileAllowed
from io import BytesIO
from base64 import b64encode
import pandas as pd
#pd.set_option('display.max_colwidth', -1)

# code which helps initialize our server
app = Flask(__name__)
app.config['SECRET_KEY'] = 'any secret key'

bootstrap = Bootstrap(app)

sales_index = ['Yes', 'No']
shelveloc_index = ['Bad', 'Good', 'Medium']
urban_index = ['Yes', 'No']
us_index = ['Yes', 'No']

# load the model from disk
model = pickle.load(open('model/model.pkl', 'rb'))
feature_names= ['CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'ShelveLoc', 'Age', 'Education', 'Urban', 'US']

class FeaturesForm(FlaskForm):
	CompPrice = FloatField('CompPrice', [validators.DataRequired(), validators.NumberRange(min=0, max=1000)])
	Income = FloatField('Income', [validators.DataRequired(), validators.NumberRange(min=0, max=10000)])
	Advertising = FloatField('Advertising', [validators.DataRequired(), validators.NumberRange(min=0, max=1000)])
	Population = FloatField('Population', [validators.DataRequired(), validators.NumberRange(min=0, max=10000)])
	Price = FloatField('Price', [validators.DataRequired(), validators.NumberRange(min=0, max=10000)])
	ShelveLoc = SelectField('ShelveLoc', [validators.DataRequired()], choices=[('Bad','Bad'), ('Good','Good'), ('Medium','Medium')])
	Age = FloatField('Age', [validators.DataRequired(), validators.NumberRange(min=1, max=100)])
	Education = FloatField('Education', [validators.DataRequired(), validators.NumberRange(min=0, max=30)])
	Urban = SelectField('Urban', [validators.DataRequired()], choices=[('Yes','Yes'), ('No','No')])
	US = SelectField('US', [validators.DataRequired()], choices=[('Yes','Yes'), ('No','No')])
	submit = SubmitField('Submit')

@app.route('/', methods=['GET','POST'])
def predict():
	form = FeaturesForm()
	if form.validate_on_submit():
		CompPrice = form.CompPrice.data
		Income = form.Income.data
		Advertising = form.Advertising.data
		Population = form.Population.data
		Price = form.Price.data
		ShelveLoc = form.ShelveLoc.data
		ShelveLoc_val = shelveloc_index.index(ShelveLoc)
		Age = form.Age.data
		Education = form.Education.data
		Urban = form.Urban.data
		Urban_val = urban_index.index(Urban)
		US = form.US.data
		US_val = us_index.index(US)
		features = [CompPrice, Income, Advertising, Population, Price, ShelveLoc, Age, Education, Urban, US]
		features_val = [CompPrice, Income, Advertising, Population, Price, ShelveLoc_val, Age, Education, Urban_val, US_val]
		df = pd.DataFrame([features], columns=feature_names)
		prediction = model.predict([features_val])
		result = sales_index[prediction[0]]
		return render_template('result.html', df = df, result=result)

	return render_template('index.html', form=form)

class ImageForm(FlaskForm):
	image = FileField('Upload an image',validators=[FileAllowed(['jpg', 'png', 'jpeg'], u'Image only!'), FileRequired(u'File was empty!')])
	submit = SubmitField('Submit')

if __name__ == '__main__':
	app.run(debug=True)
