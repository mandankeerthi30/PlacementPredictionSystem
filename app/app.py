from flask import Flask, render_template
from src.data.load_data import load_data, get_summary

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/dataset')
def dataset():
    df = load_data()
    summary = get_summary(df)

    return render_template(
        "load_dataset.html",
        summary=summary,
        first_rows=df.head().to_html(index=False)
    )


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


@app.route('/evalution')
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