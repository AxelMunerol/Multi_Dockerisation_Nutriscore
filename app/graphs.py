import io
import base64
from matplotlib.figure import Figure

def create_graph():
    # Créer un graphe avec Matplotlib
    fig = Figure()
    ax = fig.subplots()
    ax.plot([1, 2, 3, 4], [10, 20, 25, 30])
    ax.set_title("Exemple de graphique")

    # Convertir en image pour l'afficher dans le template
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')
