from flask import Flask, render_template
import tensorflow as tf

app = Flask(__name__)

# Initialize TensorFlow logging
tf.get_logger().setLevel('INFO')  # Set log level to INFO

# Initialize TensorFlow summary writer
summary_writer = tf.summary.create_file_writer('logs')

@app.route('/')
def index():
    # Set TensorFlow step
    tf.summary.experimental.set_step(1)
    # Example TensorFlow logging
    tf.summary.scalar('example_metric', 0.5)  # Log a scalar value
    return render_template('index.html')

@app.route('/about')
def about():
    # Set TensorFlow step
    tf.summary.experimental.set_step(2)
    # Example TensorFlow logging
    tf.summary.scalar('example_metric', 0.8)  # Log a scalar value
    return render_template('about.html')

@app.route('/services')
def services():
    # Set TensorFlow step
    tf.summary.experimental.set_step(3)
    # Example TensorFlow logging
    tf.summary.scalar('example_metric', 1.2)  # Log a scalar value
    return render_template('services.html')

@app.route('/contact')
def contact():
    # Set TensorFlow step
    tf.summary.experimental.set_step(4)
    # Example TensorFlow logging
    tf.summary.scalar('example_metric', 1.5)  # Log a scalar value
    return render_template('contact.html')

if __name__ == '__main__':
    with summary_writer.as_default():
        app.run(threaded=False, processes=1)
