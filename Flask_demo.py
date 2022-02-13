from flask import Flask,render_template,jsonify,request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return 'Hello World'

@app.route("/home")
def home():
    return render_template("form.html")

@app.route("/api",methods=['GET'])
def printString():
    dict={}
    dict['name']=str(request.args['name'])
    return jsonify(dict)

#http://127.0.0.1:5000/api?name=rohan
      
if __name__=="__main__":
     app.run(debug=True)
