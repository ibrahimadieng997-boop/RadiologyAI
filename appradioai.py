import json, os
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import assets, core
from author_profile import PROFILE as PF

st.set_page_config(page_title="RadiologyAI", page_icon="🫁", layout="wide", initial_sidebar_state="collapsed")
try:  # desactive la traduction auto du navigateur (evite l'erreur removeChild) ; ignore si l'API change
    import streamlit.components.v1 as components
    components.html("""<script>
const d = window.parent.document;
d.documentElement.setAttribute('translate', 'no'); d.documentElement.lang = 'fr';
const m = d.createElement('meta'); m.name = 'google'; m.content = 'notranslate'; d.head.appendChild(m);
</script>""", height=0)
except Exception:
    pass

# ------------------------------------------------------------------ style
CSS = """
<style>
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stSidebar"]{display:none}
.stApp{background:radial-gradient(1200px 600px at 80% -10%,#0b2a40 0,#070d1a 55%)}
.block-container{padding:0 2.2rem 3rem;max-width:1500px}
html,body,[class*="css"]{font-family:'Inter','Segoe UI',sans-serif;color:#e2e8f0}
.hero{position:relative;height:300px;margin:0 -2.2rem;background-size:cover;background-position:right center;
 display:flex;flex-direction:column;justify-content:center;padding:0 3.5rem;border-bottom:1px solid #164e63}
.hero h1{font-size:3.6rem;font-weight:800;margin:0;letter-spacing:-1.5px;
 background:linear-gradient(90deg,#fff,#67e8f9 60%,#34d399);-webkit-background-clip:text;color:transparent}
.hero p{color:#94a3b8;font-size:1.15rem;margin:.3rem 0 1rem}
.chip{display:inline-block;padding:.25rem .8rem;border:1px solid #155e75;border-radius:99px;font-size:.78rem;
 color:#a5f3fc;background:#0891b21f;margin-right:.5rem}
div[role="radiogroup"]{display:flex;gap:.6rem;width:100%;background:#0b1424e6;border:1px solid #1e3a5f;
 border-radius:16px;padding:.55rem;margin:-1.7rem 0 1.6rem;position:relative;z-index:5;box-shadow:0 12px 40px #0008;backdrop-filter:blur(10px)}
div[role="radiogroup"]>label{flex:1;justify-content:center;background:#0f1d33;border:1px solid #1e3a5f;border-radius:11px;
 padding:.85rem .4rem;cursor:pointer;transition:.2s}
div[role="radiogroup"]>label>div:first-child{display:none}
div[role="radiogroup"]>label:hover{border-color:#22d3ee;transform:translateY(-2px)}
div[role="radiogroup"]>label:has(input:checked){background:linear-gradient(135deg,#0891b2,#059669);border-color:#67e8f9;box-shadow:0 0 22px #22d3ee55}
div[role="radiogroup"] p{font-weight:700;font-size:1rem;text-align:center;margin:0}
.card{background:linear-gradient(160deg,#0f1d33,#0a1424);border:1px solid #1e3a5f;border-radius:16px;padding:1.2rem 1.3rem;margin-bottom:1rem}
.kpi{text-align:center}.kpi b{display:block;font-size:2.3rem;background:linear-gradient(90deg,#67e8f9,#34d399);-webkit-background-clip:text;color:transparent}
.kpi span{color:#94a3b8;font-size:.82rem;text-transform:uppercase;letter-spacing:1px}
.badge{display:inline-block;padding:.35rem 1rem;border-radius:99px;font-weight:800;letter-spacing:.5px}
.ok{background:#05966922;color:#34d399;border:1px solid #059669}.ko{background:#e11d4822;color:#fb7185;border:1px solid #e11d48}
.wait{background:#d9770622;color:#fbbf24;border:1px solid #d97706}
.xr{border-radius:12px;width:100%;border:1px solid #1e3a5f}
.bar{height:9px;border-radius:9px;background:#1e293b;overflow:hidden;margin-top:.45rem}.bar i{display:block;height:100%}
h2,h3{color:#f1f5f9!important}
.stButton>button,.stDownloadButton>button{border-radius:11px;font-weight:700;padding:.65rem 1.2rem;border:1px solid #22d3ee;background:#0891b2;color:#fff}
.stButton>button:hover{background:#06b6d4;border-color:#a5f3fc}
.avatar{width:210px;height:210px;border-radius:50%;object-fit:cover;object-position:50% 20%;border:3px solid #22d3ee;
 box-shadow:0 0 0 7px #22d3ee22,0 0 45px #22d3ee66}
.tag{display:inline-block;padding:.28rem .75rem;margin:.18rem;border-radius:99px;font-size:.82rem;color:#a5f3fc;background:#0891b21f;border:1px solid #155e75}
.tl{border-left:2px solid #155e75;padding:0 0 .9rem 1rem;margin-left:.3rem}.tl small{color:#67e8f9}
.btnlink{display:inline-block;padding:.5rem 1.1rem;margin:.25rem;border-radius:10px;background:#0891b2;color:#fff!important;font-weight:700;text-decoration:none}
.foot{text-align:center;color:#64748b;font-size:.82rem;margin-top:2rem;padding-top:1rem;border-top:1px solid #1e3a5f}
.warn{background:#d977061f;border:1px solid #d97706;border-radius:12px;padding:.7rem 1rem;color:#fcd34d;margin-bottom:1rem}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

@st.cache_resource(show_spinner="Chargement du modele VGG16...")
def get_bundle(): return core.load_model()
bundle = get_bundle()

def jload(name):
    p = f"{core.ART}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None

# ------------------------------------------------------------------ donnees (reelles ou demo)
def get_metrics():
    m = jload("metrics.json")
    if m: return m, False
    k = 32; f = np.linspace(0, 1, 60); r = np.linspace(0, 1, 60)
    return {"accuracy": .9206, "loss": .1842, "n_test": 63, "precision": [.91, .93], "recall": [.88, .95], "f1": [.89, .94],
            "auc": .97, "ap": .98, "roc": {"fpr": f.tolist(), "tpr": (1 - (1 - f) ** k).tolist()},
            "pr": {"recall": r.tolist(), "precision": (1 - .07 * r ** 6).tolist()}, "cm": [[21, 3], [2, 37]]}, True

def get_history():
    h = jload("history.json")
    if h: return h, False
    e = np.arange(14); vl = .5 + .6 * np.exp(-e / 2) - .05 * (1 - np.exp(-e / 4)) - e * .02
    return {"loss": (2.2 * np.exp(-e / 5) + .25).tolist(), "val_loss": vl.tolist(),
            "accuracy": (.93 - .38 * np.exp(-e / 4.5)).tolist(), "val_accuracy": (.88 - .21 * np.exp(-e / 3)).tolist()}, True

def get_samples():
    s = jload("samples.json")
    if s: return [(Image.open(f"{core.ART}/{x['file']}").convert("RGB"), x["true"], x["prob"]) for x in s], False
    out = []
    for i in range(4):
        out.append((assets.demo_xray("normal", i), 0, [.02, .05, .11, .31][i]))
        out.append((assets.demo_xray("pneumonia", 10 + i), 1, [.97, .91, .78, .64][i]))
    return out, True

DARK = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,29,51,.6)",
            margin=dict(l=40, r=20, t=50, b=40), font=dict(family="Inter"))
def demo_note(flag):
    if flag: st.markdown('<div class="warn">Donnees de <b>demonstration</b> (artifacts/ absent) : lancez <code>export()</code> depuis votre notebook pour afficher vos vraies metriques.</div>', unsafe_allow_html=True)

def badge(status):
    c = {"VALIDE": "ok", "NON VALIDE": "ko", "EN ATTENTE": "wait"}[status]
    t = {"VALIDE": "✔ VALIDÉ PAR LE MÉDECIN", "NON VALIDE": "✖ NON VALIDÉ", "EN ATTENTE": "⏳ EN ATTENTE DE VALIDATION"}[status]
    return f'<span class="badge {c}">{t}</span>'

def gauge(p):
    col = "#fb7185" if p >= .5 else "#34d399"
    f = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number=dict(suffix="%", font=dict(size=46)),
        title=dict(text="Probabilité de pneumonie"),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=col, thickness=.3), bgcolor="#0f1d33",
                   steps=[dict(range=[0, 35], color="#064e3b"), dict(range=[35, 65], color="#78350f"), dict(range=[65, 100], color="#7f1d1d")],
                   threshold=dict(line=dict(color="white", width=3), value=50))))
    f.update_layout(height=290, **{**DARK, "margin": dict(l=30, r=30, t=60, b=10)}); return f

def show_case(case, key=""):
    d = f"{core.DATA}/cases/{case['id']}"
    a, b, c = st.columns([1, 1, 1])
    a.markdown("**Radiographie originale**"); a.image(f"{d}/original.png", width="stretch")
    b.markdown("**Grad-CAM : zones d'attention**"); b.image(f"{d}/gradcam.png", width="stretch")
    c.plotly_chart(gauge(case["prob"]), width="stretch", key="g" + key + case["id"])
    c.markdown(f"<div style='text-align:center'><span class='badge {'ko' if case['label']=='PNEUMONIA' else 'ok'}'>{case['label']}</span>"
               f"<br><small style='color:#94a3b8'>Confiance {case['confiance']*100:.1f}% · {case['niveau']}</small></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ en-tete + navigation
st.markdown(f"""<div class="hero" translate="no" style="background-image:url('{assets.hero_uri()}')">
<h1>RadiologyAI</h1><p>Aide au diagnostic de la pneumonie par intelligence artificielle · validation médicale intégrée</p>
<div><span class="chip">VGG16 · Transfer Learning</span><span class="chip">Grad-CAM explicable</span><span class="chip">Rapport PDF avec validation médecin</span></div></div>""", unsafe_allow_html=True)
PAGES = ["🏠 Accueil", "🔬 Analyse patient", "🩺 Validation médicale", "🖼️ Galerie", "📊 Performances", "📄 Dossiers & rapports", "👤 Auteur"]
page = st.radio("nav", PAGES, horizontal=True, label_visibility="collapsed")
if bundle is None:
    st.markdown('<div class="warn">⚠ <b>Mode démonstration</b> : <code>artifacts/model.keras</code> introuvable. Les scores sont <b>simulés</b> et sans valeur clinique.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ ACCUEIL
if page == PAGES[0]:
    m, dm = get_metrics(); cases = core.list_cases()
    k = st.columns(5)
    vals = [(f"{m['accuracy']*100:.1f}%", "Accuracy test"), (f"{m['recall'][1]*100:.0f}%", "Sensibilité pneumonie"),
            (f"{m['f1'][1]:.2f}", "F1-score"), (f"{m['auc']:.2f}", "AUC ROC"), (str(len(cases)), "Dossiers traités")]
    for col, (v, l) in zip(k, vals): col.markdown(f'<div class="card kpi"><b>{v}</b><span>{l}</span></div>', unsafe_allow_html=True)
    demo_note(dm)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""<div class="card"><h3>Un flux clinique complet, en 4 étapes</h3>
        <p><b>1 · Import</b> de la radiographie et saisie du dossier patient.<br>
        <b>2 · Analyse IA</b> : probabilité de pneumonie et carte Grad-CAM montrant <i>où</i> le modèle regarde.<br>
        <b>3 · Validation</b> par le médecin : valider ou contester, avec diagnostic retenu et commentaire.<br>
        <b>4 · Rapport PDF</b> signé, avec fiche résultat patient et recommandations.</p></div>
        <div class="card"><h3>Pourquoi cette application est différente</h3>
        <p>✔ Le médecin reste décisionnaire : aucun résultat n'est final sans sa validation.<br>
        ✔ Explicabilité Grad-CAM pour chaque patient, pas une simple boîte noire.<br>
        ✔ Zone d'incertitude signalée automatiquement (35–65 %) avec relecture recommandée.<br>
        ✔ Traçabilité : dossier, date, médecin, décision et commentaire dans chaque rapport.</p></div>""", unsafe_allow_html=True)
    with c2:
        sm, _ = get_samples()
        n = next(s for s in sm if s[1] == 0); p = next(s for s in sm if s[1] == 1)
        st.markdown(f'<div class="card"><h3>Aperçu</h3><div style="display:flex;gap:.8rem">'
                    f'<div><img class="xr" src="data:image/png;base64,{assets.b64(n[0])}"><center><small>Normal</small></center></div>'
                    f'<div><img class="xr" src="data:image/png;base64,{assets.b64(p[0])}"><center><small>Pneumonie</small></center></div></div></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ ANALYSE
elif page == PAGES[1]:
    st.subheader("Nouvelle analyse")
    l, r = st.columns([1, 1.3])
    with l:
        with st.form("patient"):
            nom = st.text_input("Nom et prénom du patient"); c1, c2 = st.columns(2)
            age = c1.number_input("Âge", 0, 120, 45); sexe = c2.selectbox("Sexe", ["Homme", "Femme"])
            pid = st.text_input("Identifiant patient"); sym = st.text_area("Symptômes / contexte clinique", height=90)
            up = st.file_uploader("Radiographie thoracique (PNG / JPG)", type=["png", "jpg", "jpeg"])
            go_ = st.form_submit_button("🔬 Lancer l'analyse IA", width="stretch")
    if go_:
        if not up or not nom.strip(): st.error("Renseignez le nom du patient et importez une radiographie.")
        else:
            with st.spinner("Analyse en cours..."):
                pil = Image.open(up); prob, heat, demo = core.analyse(pil, bundle)
                case = core.save_case(dict(nom=nom, age=age, sexe=sexe, pid=pid, symptomes=sym), pil, core.overlay(pil, heat), prob, demo)
            st.session_state.current = case["id"]
    cur = next((c for c in core.list_cases() if c["id"] == st.session_state.get("current")), None)
    with r:
        if cur:
            st.markdown(f"### Dossier {cur['id']} — {cur['patient']['nom']}"); st.markdown(badge(cur["validation"]["status"]), unsafe_allow_html=True)
        else: st.info("Remplissez le formulaire pour obtenir l'analyse.")
    if cur:
        show_case(cur, "an"); st.markdown("### Recommandations")
        for x in core.recommandations(cur["prob"]): st.markdown(f"- {x}")
        st.success("Dossier enregistré. Rendez-vous dans « Validation médicale » pour la décision du médecin.")

# ------------------------------------------------------------------ VALIDATION
elif page == PAGES[2]:
    st.subheader("Validation par le médecin")
    cases = core.list_cases()
    if not cases: st.info("Aucun dossier. Lancez d'abord une analyse.")
    else:
        opts = {f"{c['id']} · {c['patient']['nom']} · {c['validation']['status']}": c for c in cases}
        sel = opts[st.selectbox("Dossier à examiner", list(opts))]; case = sel
        st.markdown(badge(case["validation"]["status"]), unsafe_allow_html=True)
        p = case["patient"]; st.caption(f"{p['nom']} · {p['age']} ans · {p['sexe']} · ID {p['pid'] or '-'} · {p['symptomes'] or 'sans symptômes renseignés'}")
        show_case(case, "va")
        st.markdown("#### Décision médicale")
        med = st.text_input("Nom du médecin", key="med" + case["id"]); com = st.text_area("Commentaire clinique", key="com" + case["id"])
        final = st.selectbox("Diagnostic retenu (si désaccord avec l'IA)", ["NORMAL", "PNEUMONIA"], index=0 if case["label"] == "NORMAL" else 1, key="fin" + case["id"])
        b1, b2 = st.columns(2)
        if b1.button("✅ VALIDER le diagnostic IA", width="stretch"):
            if not med.strip(): st.error("Indiquez le nom du médecin.")
            else: core.update_validation(case, "VALIDE", med, com, case["label"]); st.rerun()
        if b2.button("❌ NE PAS VALIDER (désaccord)", width="stretch"):
            if not med.strip(): st.error("Indiquez le nom du médecin.")
            else: core.update_validation(case, "NON VALIDE", med, com, final); st.rerun()

# ------------------------------------------------------------------ GALERIE
elif page == PAGES[3]:
    st.subheader("Galerie comparative : radiographies normales vs pneumonie")
    smp, dm = get_samples(); demo_note(dm)
    cols = st.columns(2)
    for col, (cls, title, color) in zip(cols, [(0, "Radiographies NORMALES", "#34d399"), (1, "Radiographies PNEUMONIE", "#fb7185")]):
        col.markdown(f"<h3 style='color:{color}!important'>{title}</h3>", unsafe_allow_html=True)
        items = [s for s in smp if s[1] == cls][:6]; g = col.columns(2)
        for i, (im, _, pr) in enumerate(items):
            g[i % 2].markdown(f"""<div class="card" style="padding:.7rem"><img class="xr" src="data:image/png;base64,{assets.b64(im)}">
            <div style="display:flex;justify-content:space-between;margin-top:.5rem"><span>Score pneumonie</span><b style="color:{'#fb7185' if pr>=.5 else '#34d399'}">{pr*100:.1f}%</b></div>
            <div class="bar"><i style="width:{pr*100:.0f}%;background:{'linear-gradient(90deg,#f59e0b,#e11d48)' if pr>=.5 else 'linear-gradient(90deg,#059669,#34d399)'}"></i></div>
            <small style="color:#94a3b8">Vérité terrain : {'PNEUMONIA' if cls else 'NORMAL'} · prédit : {'PNEUMONIA' if pr>=.5 else 'NORMAL'}</small></div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ PERFORMANCES
elif page == PAGES[4]:
    st.subheader("Performances du modèle VGG16"); m, dm = get_metrics(); h, dh = get_history(); demo_note(dm or dh)
    k = st.columns(6)
    for col, (v, l) in zip(k, [(f"{m['accuracy']*100:.1f}%", "Accuracy"), (f"{m['precision'][1]:.2f}", "Précision"), (f"{m['recall'][1]:.2f}", "Rappel"),
                               (f"{m['f1'][1]:.2f}", "F1-score"), (f"{m['auc']:.3f}", "AUC ROC"), (f"{m['ap']:.3f}", "AP (PR)")]):
        col.markdown(f'<div class="card kpi"><b>{v}</b><span>{l}</span></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    f = go.Figure(); f.add_scatter(x=m["roc"]["fpr"], y=m["roc"]["tpr"], fill="tozeroy", line=dict(color="#22d3ee", width=3), name="VGG16", fillcolor="rgba(34,211,238,.15)")
    f.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="#64748b"), name="Hasard")
    f.update_layout(title=f"Courbe ROC (AUC = {m['auc']:.3f})", xaxis_title="Taux de faux positifs", yaxis_title="Taux de vrais positifs", height=400, **DARK); c1.plotly_chart(f, width="stretch")
    f = go.Figure(go.Scatter(x=m["pr"]["recall"], y=m["pr"]["precision"], fill="tozeroy", line=dict(color="#34d399", width=3), fillcolor="rgba(52,211,153,.15)"))
    f.update_layout(title=f"Courbe Précision-Rappel (AP = {m['ap']:.3f})", xaxis_title="Rappel", yaxis_title="Précision", yaxis_range=[0, 1.02], height=400, **DARK); c2.plotly_chart(f, width="stretch")
    c3, c4 = st.columns(2); ep = list(range(1, len(h["loss"]) + 1))
    f = go.Figure(); f.add_scatter(x=ep, y=h["loss"], name="Train loss", line=dict(color="#22d3ee", width=3)); f.add_scatter(x=ep, y=h["val_loss"], name="Val loss", line=dict(color="#f59e0b", width=3))
    f.update_layout(title="Courbe de loss", xaxis_title="Époque", height=380, **DARK); c3.plotly_chart(f, width="stretch")
    f = go.Figure(); f.add_scatter(x=ep, y=h["accuracy"], name="Train acc", line=dict(color="#22d3ee", width=3)); f.add_scatter(x=ep, y=h["val_accuracy"], name="Val acc", line=dict(color="#f59e0b", width=3))
    f.update_layout(title="Courbe d'accuracy", xaxis_title="Époque", height=380, **DARK); c4.plotly_chart(f, width="stretch")
    c5, c6 = st.columns(2); cm = np.array(m["cm"])
    f = go.Figure(go.Heatmap(z=cm, x=["NORMAL", "PNEUMONIA"], y=["NORMAL", "PNEUMONIA"], text=cm, texttemplate="%{text}", textfont=dict(size=30), colorscale="Teal", showscale=False))
    f.update_layout(title="Matrice de confusion", xaxis_title="Prédit", yaxis_title="Réel", yaxis_autorange="reversed", height=380, **DARK); c5.plotly_chart(f, width="stretch")
    f = go.Figure(); 
    for n, key, c in [("Précision", "precision", "#22d3ee"), ("Rappel", "recall", "#34d399"), ("F1-score", "f1", "#f59e0b")]:
        f.add_bar(x=["NORMAL", "PNEUMONIA"], y=m[key], name=n, marker_color=c)
    f.update_layout(title="Scores par classe", barmode="group", yaxis_range=[0, 1], height=380, **DARK); c6.plotly_chart(f, width="stretch")

# ------------------------------------------------------------------ AUTEUR
elif page == PAGES[6]:
    from PIL import Image as _I
    ph = assets.b64(_I.open("media/profile.jpg").convert("RGB"), "JPEG")
    l, r = st.columns([1, 2.2])
    with l:
        tel = f"<br>📞 {PF['tel']}" if PF.get("tel") else ""
        st.markdown(f"""<div class="card" style="text-align:center"><img class="avatar" src="data:image/jpeg;base64,{ph}">
        <h2 style="margin:1rem 0 .2rem">{PF['nom']}</h2><div style="color:#67e8f9;font-weight:700">{PF['titre']}</div>
        <div style="color:#94a3b8">{PF['sous_titre']}</div><p style="color:#94a3b8;margin-top:.8rem">📍 {PF['lieu']}<br>✉️ {PF['email']}{tel}<br>🗣️ {PF['langues']}</p>
        <a class="btnlink" href="{PF['github']}" target="_blank">GitHub</a><a class="btnlink" href="mailto:{PF['email']}">Contact</a></div>""", unsafe_allow_html=True)
    with r:
        st.markdown(f'<div class="card"><h3>À propos</h3><p>{PF["resume"]}</p></div>', unsafe_allow_html=True)
        cards = "".join(f'<div class="card" style="margin:0"><b style="color:#67e8f9">{n}</b><p style="color:#cbd5e1;margin:.4rem 0 0">{d}</p></div>' for n, d in PF["projets"])
        st.markdown(f'<h3>Projets</h3><div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem">{cards}</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    sk = "".join(f"<p style='margin:.5rem 0 .1rem;color:#94a3b8'>{k}</p>" + "".join(f'<span class="tag">{t}</span>' for t in v) for k, v in PF["competences"].items())
    c1.markdown(f'<div class="card"><h3>Compétences</h3>{sk}</div>', unsafe_allow_html=True)
    tl = "".join(f'<div class="tl"><small>{a}</small><br><b>{b}</b><br><span style="color:#94a3b8">{c}</span></div>' for a, b, c in PF["parcours"])
    c2.markdown(f'<div class="card"><h3>Formation et certifications</h3>{tl}<p style="color:#94a3b8;margin-top:.6rem">🤝 {PF["engagement"]}</p></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ DOSSIERS
elif page == PAGES[5]:
    st.subheader("Dossiers patients et rapports PDF")
    cases = core.list_cases()
    if not cases: st.info("Aucun dossier enregistré.")
    else:
        st.dataframe([{"Dossier": c["id"], "Date": c["date"], "Patient": c["patient"]["nom"], "IA": c["label"], "Proba pneumonie": f"{c['prob']*100:.1f}%",
                       "Statut": c["validation"]["status"], "Médecin": c["validation"]["medecin"]} for c in cases], width="stretch", hide_index=True)
        case = {f"{c['id']} · {c['patient']['nom']}": c for c in cases}
        case = case[st.selectbox("Ouvrir la fiche", list(case))]; v = case["validation"]; p = case["patient"]
        st.markdown(f"""<div class="card"><div style="display:flex;justify-content:space-between;align-items:center"><h3 style="margin:0">Fiche résultat · {p['nom']}</h3>{badge(v['status'])}</div>
        <p style="color:#94a3b8">{p['age']} ans · {p['sexe']} · ID {p['pid'] or '-'} · Dossier {case['id']} · {case['date']}</p>
        <p><b>Résultat IA :</b> {case['label']} ({case['prob']*100:.1f}% pneumonie, confiance {case['niveau']})<br>
        <b>Médecin :</b> {v['medecin'] or '-'} {('· diagnostic retenu : <b>'+v['final']+'</b>') if v['final'] else ''}<br><b>Commentaire :</b> {v['commentaire'] or '-'}</p></div>""", unsafe_allow_html=True)
        show_case(case, "do"); st.markdown("### Recommandations")
        for x in core.recommandations(case["prob"], v["status"], v["final"]): st.markdown(f"- {x}")
        st.download_button("📥 Télécharger le rapport PDF", core.make_pdf(case), f"rapport_{case['id']}.pdf", "application/pdf")

st.markdown(f'<div class="foot">RadiologyAI · développé par {PF["nom"]} · Dakar Institute of Technology · aide à la décision, non certifié dispositif médical</div>', unsafe_allow_html=True)
