from flask import Flask, render_template, request, redirect, session, flash, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)
app.secret_key = "secret_key_dermoscan_ai"

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Créer automatiquement le dossier des téléversements s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Charger le modèle IA VGG16
MODEL_PATH = "model/vgg16_skin_cancer.h5"
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print("======== MODÈLE IA VGG16 CHARGÉ AVEC SUCCÈS ========")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
else:
    print(f"ATTENTION : Le fichier de modèle est introuvable à {MODEL_PATH}")

# Variables globales de base de données
db = None
cursor = None
db_connected = False

def init_db():
    """Tente d'initialiser la base de données et de créer les tables automatiquement."""
    global db, cursor, db_connected
    try:
        # Se connecter à MySQL sans spécifier de base de données d'abord
        temp_db = mysql.connector.connect(
            host="localhost",
            user="root",
            password=""
        )
        temp_cursor = temp_db.cursor()
        
        # Créer la base de données si elle n'existe pas
        temp_cursor.execute("CREATE DATABASE IF NOT EXISTS skin_cancer_db")
        temp_db.commit()
        temp_cursor.close()
        temp_db.close()
        
        # Se connecter à la base de données réelle
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="skin_cancer_db"
        )
        # Utiliser buffered=True pour éviter les erreurs "Unread result found"
        cursor = db.cursor(dictionary=True, buffered=True)
        
        # Créer la table des utilisateurs si elle n'existe pas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL
            )
        """)
        
        # Créer la table des patients si elle n'existe pas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                age INT NOT NULL,
                result VARCHAR(50) NOT NULL,
                probability FLOAT NOT NULL,
                image_path VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insérer le compte admin par défaut s'il n'existe pas
        cursor.execute("SELECT * FROM users WHERE username = 'admin'")
        if not cursor.fetchone():
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", ('admin', '1234'))
            db.commit()
            print("======== COMPTE ADMINISTRATEUR PAR DÉFAUT CRÉÉ ========")
            
        db_connected = True
        print("======== BASE DE DONNÉES MYSQL CONNECTÉE ET INITIALISÉE ========")
    except Error as e:
        db_connected = False
        db = None
        cursor = None
        print("\n!!! ERREUR MYSQL DE DÉMARRAGE !!!")
        print("Impossible de se connecter à MySQL. Veuillez lancer XAMPP (Apache & MySQL).")
        print(f"Détail : {e}\n")

# Essayer d'initialiser au lancement
init_db()

def verify_connection():
    """Vérifie si la connexion est active et essaie de se reconnecter en cas de besoin."""
    global db, cursor, db_connected
    if not db_connected or db is None:
        init_db()
    else:
        try:
            db.ping(reconnect=True, attempts=3, delay=1)
        except Error:
            db_connected = False
            init_db()
    return db_connected

# Injecter l'état de la connexion dans toutes les pages
@app.context_processor
def inject_db_status():
    return dict(db_connected=db_connected)

#----------- LOGIN -----------
@app.route("/", methods=["GET", "POST"])
def login():
    if not verify_connection():
        flash("La base de données n'est pas connectée. Veuillez lancer XAMPP (Apache & MySQL) !", "danger")
        
    if request.method == "POST":
        if not db_connected:
            flash("Erreur de connexion MySQL. Veuillez vérifier que XAMPP est démarré.", "danger")
            return redirect(url_for('login'))
            
        user = request.form["username"]
        pwd = request.form["password"]

        try:
            verify_connection()
            cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (user, pwd))
            result = cursor.fetchone()

            if result:
                session["user"] = user
                flash("Connexion réussie ! Bienvenue.", "success")
                return redirect(url_for('dashboard'))
            else:
                flash("Identifiant ou mot de passe incorrect.", "danger")
        except Error as e:
            flash(f"Erreur de base de données : {e}", "danger")

    return render_template("login.html")

#-------- DASHBOARD --------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for('login'))
    return render_template("dashboard.html")

#------- PREDICT --------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for('login'))
        
    if not verify_connection():
        flash("La base de données n'est pas connectée. Impossible d'enregistrer les analyses !", "danger")
        
    if request.method == "POST":
        if not db_connected:
            flash("Veuillez d'abord démarrer MySQL sur XAMPP.", "danger")
            return redirect(url_for('predict'))
            
        if model is None:
            flash("Le modèle d'IA n'est pas chargé. Impossible de réaliser la prédiction !", "danger")
            return redirect(url_for('predict'))

        try:
            name = request.form["name"]
            age = request.form["age"]
            file = request.files["image"]

            if file.filename == "":
                flash("Veuillez choisir un cliché dermatologique à analyser.", "warning")
                return redirect(url_for('predict'))
                
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)

            # Prétraitement de l'image pour VGG16 (224x224 px)
            img = image.load_img(path, target_size=(224, 224))
            img = image.img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            # Lancement de la prédiction
            pred = model.predict(img)[0][0]
            result = "Malignant" if pred > 0.5 else "Benign"
            
            # Calculer la confiance relative au diagnostic choisi
            # Si Malin (pred > 0.5), la probabilité de malignité est pred.
            # Si Bénin (pred <= 0.5), la confiance de bénignité est 1 - pred.
            confidence = pred if pred > 0.5 else (1.0 - pred)
            prob_percent = round(confidence * 100, 2)
            
            # Sauvegarder dans la base de données
            verify_connection()
            cursor.execute("""
                INSERT INTO patients (name, age, result, probability, image_path)
                VALUES (%s, %s, %s, %s, %s)
            """, (name, age, result, float(pred), path))
            db.commit()
                           
            flash("Analyse clinique effectuée avec succès !", "success")
            return render_template("result.html",
                                   result=result,
                                   prob=prob_percent,
                                   img=path)
        except Exception as e:
            flash(f"Erreur système durant l'analyse : {e}", "danger")
            return redirect(url_for('predict'))
            
    return render_template("predict.html")

#------ PATIENTS ------
@app.route("/patients")
def patients():
    if "user" not in session:
        return redirect(url_for('login'))
        
    if not verify_connection():
        flash("La base de données n'est pas connectée. Impossible d'afficher le registre !", "danger")
        return render_template("patients.html", patients=[])
        
    try:
        verify_connection()
        cursor.execute("SELECT * FROM patients ORDER BY created_at DESC")
        data = cursor.fetchall()
        return render_template("patients.html", patients=data)
    except Error as e:
        flash(f"Erreur lors de la récupération des données : {e}", "danger")
        return render_template("patients.html", patients=[])

#-------------- LOGOUT -----------------
@app.route("/logout")
def logout():
    session.clear()
    flash("Vous avez été déconnecté de votre espace clinique.", "info")
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)
