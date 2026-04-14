"""
Tests des endpoints FastAPI de l'API d'assurance auto.
Couvre : health check, predict/frequency, predict/severity, predict/premium,
         et les cas d'erreur (champs manquants, types invalides).
"""

from fastapi.testclient import TestClient

from auto_insurance.api.main import app

client = TestClient(app)

# --- Payload valide réutilisé dans tous les tests ---
VALID_PAYLOAD = {
    "type_contrat": "A",
    "duree_contrat": 12.0,
    "anciennete_info": 5.0,
    "freq_paiement": "mensuel",
    "utilisation": "prive",
    "code_postal": "75001",
    "age_conducteur1": 35.0,
    "sex_conducteur1": "M",
    "anciennete_permis1": 12.0,
    "anciennete_vehicule": 3.0,
    "cylindre_vehicule": 1600.0,
    "din_vehicule": 90.0,
    "essence_vehicule": "essence",
    "marque_vehicule": "Peugeot",
    "modele_vehicule": "308",
    "fin_vente_vehicule": 2022.0,
    "vitesse_vehicule": 180.0,
    "type_vehicule": "berline",
    "prix_vehicule": 18000.0,
    "poids_vehicule": 1200.0,
}


# ──────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────

class TestHealth:
    def test_health_status_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self):
        response = client.get("/health")
        body = response.json()
        assert "status" in body
        assert "message" in body

    def test_health_status_ok(self):
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_health_audit_endpoint_structure(self):
        response = client.get("/health/audit")
        body = response.json()
        assert response.status_code == 200
        assert body["status"] == "ok"
        assert "audit" in body
        assert "enabled" in body["audit"]

    def test_request_id_header_present(self):
        response = client.get("/health")
        assert "X-Request-ID" in response.headers


# ──────────────────────────────────────────────
# /predict/frequency
# ──────────────────────────────────────────────

class TestPredictFrequency:
    def test_frequency_status_200(self):
        response = client.post("/predict/frequency", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_frequency_has_key(self):
        response = client.post("/predict/frequency", json=VALID_PAYLOAD)
        assert "frequence_predite" in response.json()

    def test_frequency_is_float(self):
        response = client.post("/predict/frequency", json=VALID_PAYLOAD)
        assert isinstance(response.json()["frequence_predite"], float)

    def test_frequency_missing_field(self):
        payload = VALID_PAYLOAD.copy()
        del payload["age_conducteur1"]
        response = client.post("/predict/frequency", json=payload)
        assert response.status_code == 422

    def test_frequency_invalid_age(self):
        payload = {**VALID_PAYLOAD, "age_conducteur1": 10}  # < 18
        response = client.post("/predict/frequency", json=payload)
        assert response.status_code == 422


# ──────────────────────────────────────────────
# /predict/severity
# ──────────────────────────────────────────────

class TestPredictSeverity:
    def test_severity_status_200(self):
        response = client.post("/predict/severity", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_severity_has_key(self):
        response = client.post("/predict/severity", json=VALID_PAYLOAD)
        assert "cout_moyen_predit" in response.json()

    def test_severity_is_float(self):
        response = client.post("/predict/severity", json=VALID_PAYLOAD)
        assert isinstance(response.json()["cout_moyen_predit"], float)

    def test_severity_missing_field(self):
        payload = VALID_PAYLOAD.copy()
        del payload["prix_vehicule"]
        response = client.post("/predict/severity", json=payload)
        assert response.status_code == 422


# ──────────────────────────────────────────────
# /predict/premium
# ──────────────────────────────────────────────

class TestPredictPremium:
    def test_premium_status_200(self):
        response = client.post("/predict/premium", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_premium_has_all_keys(self):
        response = client.post("/predict/premium", json=VALID_PAYLOAD)
        body = response.json()
        assert "frequence_predite" in body
        assert "cout_moyen_predit" in body
        assert "prime_pure" in body

    def test_premium_formula(self):
        """prime_pure doit être cohérent avec fréquence × gravité."""
        response = client.post("/predict/premium", json=VALID_PAYLOAD)
        body = response.json()
        expected = body["frequence_predite"] * body["cout_moyen_predit"]
        assert abs(body["prime_pure"] - expected) < 1.0
    
    def test_premium_positive_values(self):
        response = client.post("/predict/premium", json=VALID_PAYLOAD)
        body = response.json()
        assert body["frequence_predite"] >= 0
        assert body["cout_moyen_predit"] >= 0
        assert body["prime_pure"] >= 0

    def test_premium_missing_field(self):
        payload = VALID_PAYLOAD.copy()
        del payload["poids_vehicule"]
        response = client.post("/predict/premium", json=payload)
        assert response.status_code == 422

    def test_premium_wrong_type(self):
        payload = {**VALID_PAYLOAD, "age_conducteur1": "pas_un_nombre"}
        response = client.post("/predict/premium", json=payload)
        assert response.status_code == 422

    def test_premium_accepts_bracketed_numeric_string(self):
        payload = {**VALID_PAYLOAD, "age_conducteur1": "[3.5E1]"}
        response = client.post("/predict/premium", json=payload)
        assert response.status_code == 200

    def test_premium_female_driver(self):
        payload = {**VALID_PAYLOAD, "sex_conducteur1": "F"}
        response = client.post("/predict/premium", json=payload)
        assert response.status_code == 200

    def test_premium_with_second_driver(self):
        payload = {**VALID_PAYLOAD, "conducteur2": "Yes"}
        response = client.post("/predict/premium", json=payload)
        assert response.status_code == 200
# ──────────────────────────────────────────────
# Validation métier
# ──────────────────────────────────────────────

class TestValidationMetier:
    def test_impossible_age_permis(self):
        """Un conducteur de 20 ans avec 15 ans de permis = impossible."""
        payload = {**VALID_PAYLOAD, "age_conducteur1": 20, "anciennete_permis1": 15}
        response = client.post("/predict/premium", json=payload)
        assert response.status_code == 422

    def test_frequence_entre_0_et_1(self):
        """La fréquence prédite doit être entre 0 et 1."""
        response = client.post("/predict/frequency", json=VALID_PAYLOAD)
        freq = response.json()["frequence_predite"]
        assert 0 <= freq <= 1

    def test_gravite_positive(self):
        """Le coût moyen prédit doit être positif."""
        response = client.post("/predict/severity", json=VALID_PAYLOAD)
        assert response.json()["cout_moyen_predit"] > 0

    def test_niveau_risque_present(self):
        """La réponse premium doit contenir un niveau de risque."""
        response = client.post("/predict/premium", json=VALID_PAYLOAD)
        assert "niveau_risque" in response.json()

    def test_niveau_risque_valide(self):
        """Le niveau de risque doit être une valeur connue."""
        response = client.post("/predict/premium", json=VALID_PAYLOAD)
        niveau = response.json()["niveau_risque"]
        assert niveau in ["faible", "modéré", "élevé", "très élevé"]
