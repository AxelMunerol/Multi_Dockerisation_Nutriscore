from flask import Flask

# Création de l'application Flask
def create_app():
    app = Flask(__name__)

    from routes import main
    app.register_blueprint(main)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)


