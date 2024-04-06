from flask import Flask, render_template
from pyrasp.pyrasp import FlaskRASP

app = Flask(__name__)

# Use configured RASP
FlaskRASP(app, conf='rasp.json')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    # Run the Flask app on a different IP address (e.g., 192.168.1.100)
    app.run(host='127.0.0.2', threaded=False, processes=1)
