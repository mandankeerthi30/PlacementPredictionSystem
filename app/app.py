from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/preprocessing')
def preprocessing():
    return render_template('preprocessing.html')

@app.route('/eda')
def eda():
    return render_template('eda.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/evalution')   # spelling matches your filename
def evalution():
    return render_template('evalution.html')

@app.route('/comparison')
def comparison():
    return render_template('comparison.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

if __name__ == '__main__':
    app.run(debug=True)