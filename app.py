from flask import Flask, render_template, request

app = Flask(__name__)
from flask import render_template

@app.route('/')
def home():
    # Load your model and compute accuracy here (or fetch it if already saved)
    accuracy = 0.92  # Replace with your actual computed accuracy
    return render_template('index.html', accuracy=accuracy)


@app.route("/")
def dashboard():
    return render_template("dashboard.html", page="dashboard")

@app.route("/alerts")
def alerts():
    return render_template("dashboard.html", page="alerts")

@app.route("/analytics")
def analytics():
    return render_template("dashboard.html", page="analytics")

@app.route("/settings", methods=["GET", "POST"])
def settings():
    # Initialize default values
    threshold = 'medium'
    alert_frequency = 5
    log_retention = 30
    
    if request.method == "POST":
        # Get form data from the request
        threshold = request.form.get("threshold", 'medium')
        alert_frequency = request.form.get("alerts", 5)
        log_retention = request.form.get("logRetention", 30)

    # Pass the values back to the template
    return render_template("dashboard.html", page="settings", threshold=threshold, alert_frequency=alert_frequency, log_retention=log_retention)

@app.route("/threat/<int:threat_id>")
def threat_detail(threat_id):
    return f"<h1>Details for Threat ID: {threat_id}</h1>"

if __name__ == "__main__":
    app.run(debug=True)
