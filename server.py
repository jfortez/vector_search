from flask import Flask, request, jsonify,  render_template
from search import get_data, search_name, search_with_ai, create_index
import json

app = Flask(__name__)

datos = get_data()
indice, nombres = create_index(datos)

@app.route("/buscar", methods=["GET"])
def buscar():
    query = request.args.get("nombre")
    if not query:
        return jsonify({"error": "Falta el parámetro 'nombre'"}), 400

    result = search_name(query, datos)
    result_ai = search_with_ai(query, indice, nombres)


    return app.response_class(
        response=json.dumps({
            "fuzzy_search": result[0] if result else None,
            "search_ai": result_ai
        }, ensure_ascii=False,indent=2),
        status=200,
        mimetype="application/json"
    )

@app.route("/datos", methods=["GET"])
def obtener_datos():
    return jsonify(datos), 200

@app.route("/buscar-nombre", methods=["GET"])
def buscar_nombre():
    query = request.args.get("nombre")
    if not query:
        return jsonify({"error": "Falta el parámetro 'nombre'"}), 400

    result = search_name(query, datos)
    return jsonify(result), 200

@app.route('/app')
def app_view():
    return render_template('index.html')

@app.route("/", methods=["GET"])
def hello_world():
    return "<h1>Hello, World!</h1>"

if __name__ == "__main__":
    app.run(debug=True, port=8080)
