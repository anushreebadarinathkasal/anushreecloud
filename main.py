# from flask import Flask
from flask import Flask, request, render_template
import os

app = Flask(__name__)

port = int(os.getenv('PORT', 6000))

@app.route('/')
def my_form():
    return render_template('my-form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    processed_text = text.upper()
    return processed_text

def hello_world():
  return 'Hello, World!\n This looks just amazing within 5 minutes'

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=port,debug=True)
