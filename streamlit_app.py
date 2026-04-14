import shap
import streamlit as st

from auto_insurance.src.pipeline import PredictionPipeline


st.set_page_config(
    page_title="AutoAssur",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


RISK_META = {
    "faible": {"label": "Faible", "color": "#1f7a4f", "accent": "#d7f5e4"},
    "modere": {"label": "Modere", "color": "#9a5b00", "accent": "#fff1cc"},
    "eleve": {"label": "Eleve", "color": "#b45309", "accent": "#ffe2bf"},
    "tres eleve": {"label": "Tres eleve", "color": "#b42318", "accent": "#ffe0db"},
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(22, 101, 52, 0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(14, 116, 144, 0.10), transparent 26%),
                linear-gradient(180deg, #f4efe6 0%, #fbfaf7 42%, #f7f9fc 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            background: linear-gradient(135deg, #16324f 0%, #245d74 55%, #2d7b61 100%);
            color: white;
            border-radius: 28px;
            padding: 2rem 2.2rem;
            box-shadow: 0 18px 60px rgba(22, 50, 79, 0.18);
            margin-bottom: 1.25rem;
        }
        .hero-kicker {
            letter-spacing: .12em;
            text-transform: uppercase;
            font-size: 0.8rem;
            opacity: 0.8;
            margin-bottom: 0.6rem;
        }
        .hero-title {
            font-size: 2.2rem;
            line-height: 1.05;
            font-weight: 800;
            margin-bottom: 0.8rem;
        }
        .hero-copy {
            font-size: 1rem;
            max-width: 48rem;
            opacity: 0.92;
        }
        .soft-card {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(22, 50, 79, 0.08);
            backdrop-filter: blur(8px);
            border-radius: 24px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 12px 40px rgba(15, 23, 42, 0.07);
            margin-bottom: 1rem;
        }
        .metric-card {
            background: white;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            min-height: 132px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        }
        .metric-label {
            color: #5b6470;
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.45rem;
        }
        .metric-value {
            color: #16324f;
            font-size: 1.9rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 0.4rem;
        }
        .metric-hint {
            color: #64748b;
            font-size: 0.92rem;
        }
        .risk-chip {
            display: inline-block;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            font-weight: 700;
            margin-top: 0.35rem;
            font-size: 0.95rem;
        }
        .section-title {
            color: #16324f;
            font-size: 1.15rem;
            font-weight: 750;
            margin-bottom: 0.85rem;
        }
        .factor-row {
            padding: 0.85rem 0;
            border-bottom: 1px solid rgba(15, 23, 42, 0.08);
        }
        .factor-row:last-child {
            border-bottom: none;
        }
        .factor-name {
            font-weight: 650;
            color: #16324f;
            margin-bottom: 0.35rem;
        }
        .factor-copy {
            color: #5b6470;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_pipeline() -> PredictionPipeline:
    return PredictionPipeline()


def get_risk_level(frequence: float) -> str:
    if frequence < 0.05:
        return "faible"
    if frequence < 0.10:
        return "modere"
    if frequence < 0.20:
        return "eleve"
    return "tres eleve"


def get_risk_factors(payload: dict, pipeline: PredictionPipeline) -> list[str]:
    df = pipeline._build_features(payload)
    df_shap = df.copy()
    for col in df_shap.select_dtypes(include="category").columns:
        df_shap[col] = df_shap[col].cat.codes

    explainer = shap.TreeExplainer(pipeline.model.model_frequence)
    shap_values = explainer.shap_values(df_shap)
    shap_importance = dict(zip(df_shap.columns.tolist(), shap_values[0]))
    top_features = sorted(
        shap_importance.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:3]

    return [
        f"{feature} {'augmente' if value > 0 else 'diminue'} le risque"
        for feature, value in top_features
    ]


def format_currency(value: float) -> str:
    return f"{value:,.2f} EUR".replace(",", " ")


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_payload(prefix: str) -> dict:
    return {
        "type_contrat": st.session_state[f"{prefix}_type_contrat"],
        "duree_contrat": float(st.session_state[f"{prefix}_duree_contrat"]),
        "anciennete_info": float(st.session_state[f"{prefix}_anciennete_info"]),
        "freq_paiement": st.session_state[f"{prefix}_freq_paiement"],
        "utilisation": st.session_state[f"{prefix}_utilisation"],
        "code_postal": str(st.session_state[f"{prefix}_code_postal"]).strip(),
        "age_conducteur1": float(st.session_state[f"{prefix}_age_conducteur1"]),
        "sex_conducteur1": st.session_state[f"{prefix}_sex_conducteur1"],
        "anciennete_permis1": float(st.session_state[f"{prefix}_anciennete_permis1"]),
        "anciennete_vehicule": float(st.session_state[f"{prefix}_anciennete_vehicule"]),
        "cylindre_vehicule": float(st.session_state[f"{prefix}_cylindre_vehicule"]),
        "din_vehicule": float(st.session_state[f"{prefix}_din_vehicule"]),
        "essence_vehicule": st.session_state[f"{prefix}_essence_vehicule"],
        "marque_vehicule": st.session_state[f"{prefix}_marque_vehicule"],
        "modele_vehicule": st.session_state[f"{prefix}_modele_vehicule"],
        "fin_vente_vehicule": float(st.session_state[f"{prefix}_fin_vente_vehicule"]),
        "vitesse_vehicule": float(st.session_state[f"{prefix}_vitesse_vehicule"]),
        "type_vehicule": st.session_state[f"{prefix}_type_vehicule"],
        "prix_vehicule": float(st.session_state[f"{prefix}_prix_vehicule"]),
        "poids_vehicule": float(st.session_state[f"{prefix}_poids_vehicule"]),
        "conducteur2": "Yes" if st.session_state.get(f"{prefix}_conducteur2", False) else "No",
    }


def render_metric_card(label: str, value: str, hint: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_card(risk_level: str, monthly_premium: float) -> None:
    meta = RISK_META[risk_level]
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Niveau de risque</div>
            <div class="metric-value" style="font-size:1.55rem;">{meta['label']}</div>
            <div class="risk-chip" style="background:{meta['accent']};color:{meta['color']};">
                Prime mensuelle estimee: {format_currency(monthly_premium)}
            </div>
            <div class="metric-hint" style="margin-top:0.7rem;">
                Lecture rapide pour la souscription et l'explication client.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_factor_panel(factors: list[str]) -> None:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Facteurs de risque dominants</div>', unsafe_allow_html=True)
    if not factors:
        st.info("Les facteurs SHAP s'afficheront apres une simulation.")
    else:
        for factor in factors:
            name = factor.split(" augmente")[0].split(" diminue")[0]
            direction = "augmente" if "augmente" in factor else "diminue"
            copy = "Tire la prime vers le haut." if direction == "augmente" else "Tire la prime vers le bas."
            st.markdown(
                f"""
                <div class="factor-row">
                    <div class="factor-name">{name}</div>
                    <div class="factor-copy">{copy}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def init_defaults(prefix: str, *, analyst: bool) -> None:
    defaults = {
        "type_contrat": "A",
        "duree_contrat": 12.0,
        "anciennete_info": 5.0,
        "freq_paiement": "mensuel",
        "utilisation": "prive",
        "code_postal": "75001",
        "age_conducteur1": 35.0 if analyst else 32.0,
        "sex_conducteur1": "M",
        "anciennete_permis1": 12.0 if analyst else 9.0,
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
        "conducteur2": False,
    }
    for field, value in defaults.items():
        key = f"{prefix}_{field}"
        if key not in st.session_state:
            st.session_state[key] = value


def render_form(prefix: str, *, analyst: bool) -> bool:
    init_defaults(prefix, analyst=analyst)
    with st.form(f"{prefix}_form", clear_on_submit=False):
        st.markdown(
            '<div class="section-title">Profil conducteur et vehicule</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input(
                "Age conducteur",
                min_value=18.0,
                max_value=100.0,
                key=f"{prefix}_age_conducteur1",
            )
            st.number_input(
                "Anciennete permis",
                min_value=0.0,
                key=f"{prefix}_anciennete_permis1",
            )
            st.text_input("Code postal", key=f"{prefix}_code_postal")
            st.selectbox("Sexe", ["M", "F"], key=f"{prefix}_sex_conducteur1")

        with col2:
            st.selectbox(
                "Marque",
                ["Peugeot", "Renault", "Citroen", "BMW", "Mercedes", "Toyota", "Audi"],
                key=f"{prefix}_marque_vehicule",
            )
            st.text_input("Modele", key=f"{prefix}_modele_vehicule")
            st.selectbox(
                "Type vehicule",
                ["berline", "suv", "citadine", "break", "coupe"],
                key=f"{prefix}_type_vehicule",
            )
            st.selectbox(
                "Carburant",
                ["essence", "diesel", "electrique", "hybride"],
                key=f"{prefix}_essence_vehicule",
            )

        with col3:
            st.number_input("Prix vehicule", min_value=0.0, key=f"{prefix}_prix_vehicule")
            st.number_input("Puissance DIN", min_value=0.0, key=f"{prefix}_din_vehicule")
            st.number_input("Cylindree", min_value=0.0, key=f"{prefix}_cylindre_vehicule")
            st.number_input("Poids", min_value=0.0, key=f"{prefix}_poids_vehicule")

        col4, col5, col6 = st.columns(3)
        with col4:
            st.number_input(
                "Vitesse max",
                min_value=0.0,
                key=f"{prefix}_vitesse_vehicule",
            )
            st.number_input(
                "Anciennete vehicule",
                min_value=0.0,
                key=f"{prefix}_anciennete_vehicule",
            )
            st.number_input(
                "Fin de vente",
                min_value=1980.0,
                max_value=2100.0,
                key=f"{prefix}_fin_vente_vehicule",
            )

        with col5:
            st.selectbox(
                "Type contrat",
                ["A", "B", "C"],
                key=f"{prefix}_type_contrat",
            )
            st.selectbox(
                "Utilisation",
                ["prive", "pro", "mixte"],
                key=f"{prefix}_utilisation",
            )
            st.selectbox(
                "Paiement",
                ["mensuel", "trimestriel", "annuel"],
                key=f"{prefix}_freq_paiement",
            )

        with col6:
            st.number_input(
                "Duree contrat (mois)",
                min_value=1.0,
                key=f"{prefix}_duree_contrat",
            )
            st.number_input(
                "Anciennete client",
                min_value=0.0,
                key=f"{prefix}_anciennete_info",
            )
            st.checkbox("Second conducteur", key=f"{prefix}_conducteur2")

        label = "Calculer le devis" if not analyst else "Lancer l'analyse complete"
        return st.form_submit_button(label, use_container_width=True)


def run_prediction(prefix: str) -> dict | None:
    try:
        payload = build_payload(prefix)
        pipeline = get_pipeline()
        result = pipeline.predict_prime(payload)
        risk_level = get_risk_level(result["frequence_predite"])
        factors = get_risk_factors(payload, pipeline)
        return {
            **result,
            "niveau_risque": risk_level,
            "facteurs_de_risque": factors,
        }
    except Exception as exc:
        st.error(f"Impossible de calculer le devis: {exc}")
        return None


def render_results(result: dict | None, *, analyst: bool) -> None:
    if result is None:
        st.info("Lance une simulation pour afficher la prime, le niveau de risque et les explications.")
        return

    monthly_premium = result["prime_pure"] / 12
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card(
            "Prime pure annuelle",
            format_currency(result["prime_pure"]),
            "Base technique avant frais, taxes et marge commerciale.",
        )
    with col2:
        render_metric_card(
            "Frequence predite",
            format_percent(result["frequence_predite"]),
            "Probabilite estimee de survenance d'un sinistre.",
        )
    with col3:
        render_metric_card(
            "Cout moyen predit",
            format_currency(result["cout_moyen_predit"]),
            "Montant moyen attendu si un sinistre survient.",
        )

    col4, col5 = st.columns([1, 1.3])
    with col4:
        render_risk_card(result["niveau_risque"], monthly_premium)
    with col5:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Lecture rapide</div>', unsafe_allow_html=True)
        st.write(
            "La prime pure suit la formule actuarielle classique:"
            f" {format_percent(result['frequence_predite'])} x {format_currency(result['cout_moyen_predit'])}."
        )
        if analyst:
            st.caption("Vue analyste: utile pour challenger un dossier et expliquer la cotation.")
        else:
            st.caption("Vue client: lecture simplifiee pour comprendre le devis estime.")
        st.markdown("</div>", unsafe_allow_html=True)

    render_factor_panel(result["facteurs_de_risque"])


def main() -> None:
    inject_styles()
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">AutoAssur • Tarification automobile</div>
            <div class="hero-title">Une interface Streamlit plus claire pour simuler, expliquer et piloter la prime pure.</div>
            <div class="hero-copy">
                Meme moteur de prediction, experience plus lisible: un parcours client pour le devis
                et un espace analyste pour la lecture actuarielle et les facteurs de risque.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## AutoAssur")
        st.write("Interface Streamlit connectee directement au pipeline de prediction.")
        st.success("Modele charge localement")
        st.caption("Deploiement Render: le health check Streamlit utilise /_stcore/health.")

    client_tab, analyst_tab = st.tabs(["Parcours client", "Vue analyste"])

    with client_tab:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Obtenir un devis en quelques champs</div>', unsafe_allow_html=True)
        submitted = render_form("client", analyst=False)
        st.markdown("</div>", unsafe_allow_html=True)
        if submitted:
            st.session_state["client_result"] = run_prediction("client")
        render_results(st.session_state.get("client_result"), analyst=False)

    with analyst_tab:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Analyse detaillee du dossier</div>', unsafe_allow_html=True)
        submitted = render_form("analyst", analyst=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if submitted:
            st.session_state["analyst_result"] = run_prediction("analyst")
        render_results(st.session_state.get("analyst_result"), analyst=True)


if __name__ == "__main__":
    main()
