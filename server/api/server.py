"""
Main module of the server file
"""

from flask import render_template
import connexion
from flask_cors import CORS


# create the application instance
app = connexion.App(__name__, specification_dir="./")

# Cead the swagger.yml file to configure the endpoints
app.add_api("swagger.yml")

CORS(app.app, resources={r"/*": {"origins": "*"}})

# Create a URL route in our application for "/"
@app.route("/")
def home():
    """
    This function just responds to the browser URL
    localhost:5000/

    :return:        the rendered templates "home.html"
    """
    return render_template("home.html")


if __name__ == "__main__":
    try:
        app.run(debug=False, threaded=False)
    except KeyboardInterrupt:
        pass