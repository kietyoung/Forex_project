from flask import Flask, request, Response
import dev_data.script as controller

app = Flask(__name__)

@app.route('/')
def hello():
    return {'hello': 'world'}


@app.route('/prices')
def get_prices():
  args = request.args
  from_date=request.args.get('from')
  pair = args['pair']
  data = controller.get_price(pair,from_date)
  return Response(data, mimetype="application/json", status=200) 
  

@app.route('/predictions')
def get_predictions():
  args = request.args
  pair = args['pair']
  model = args['model']
  steps = args['steps']
  data = controller.get_prediction(pair,model,int(steps))
  return Response(data, mimetype="application/json", status=200) 
   

if __name__ == '__main__':
     controller.update_data()
     app.run()