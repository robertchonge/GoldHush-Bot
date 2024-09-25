# Create a Procfile for Heroku
with open("Procfile", "w") as f:
    f.write("web: gunicorn main:app")  # replace 'app' with your Flask app file name without the '.py'
