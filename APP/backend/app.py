from flask import Flask, render_template, request, jsonify
#创建Flask对象app并初始化
app = Flask(__name__)


@app.route("/coldstart/user",methods=["GET", "POST"])
def userInfo():
    if request.method == 'POST':
        count = request.form.get("name")
        age = request.form.get("age")
        return(render_template('index.html',
                               settings = {'showVote': True, 'people': count, 'buttonDisable': True, 'recommendation': None}))

    if request.method == "GET":
        name = request.args.get("name")
        age = request.args.get("age")
        return name

@app.route("/coldstart/rate",methods=["GET", "POST"])
def userRate():
    if request.method == 'POST':
        count = request.form.get("name")
        age = request.form.get("age")
        return(render_template('index.html',
                               settings = {'showVote': True, 'people': count, 'buttonDisable': True, 'recommendation': None}))

    if request.method == "GET":
        name = request.args.get("name")
        age = request.args.get("age")
        return name

@app.route("/recommend",methods=["GET", "POST"])
#从这里定义具体的函数 返回值均为json格式
def movieRecommend():
    if request.method == 'POST':
        #由于POST、GET获取数据的方式不同，需要使用if语句进行判断
        count = request.form.get("name")
        age = request.form.get("age")
        return(render_template('index.html',
                               settings = {'showVote': True, 'people': count, 'buttonDisable': True, 'recommendation': None}))

    if request.method == "GET":
        name = request.args.get("name")
        age = request.args.get("age")
        return name


if __name__ == '__main__':
    app.run(port=8080)
