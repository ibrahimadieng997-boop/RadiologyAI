"""Moteur : inference VGG16, Grad-CAM, recommandations, dossiers patients, rapport PDF."""
import io, json, os, uuid
from datetime import datetime
import numpy as np
from PIL import Image

ART, DATA = "artifacts", "data"
os.makedirs(f"{DATA}/cases", exist_ok=True)

# ---------------------------------------------------------------- modele
def load_model():
    p = f"{ART}/model.keras"
    if not os.path.exists(p):
        return None
    from tensorflow import keras
    from tensorflow.keras.layers import Input
    model = keras.models.load_model(p)
    vgg = model.get_layer("vgg16")                       # meme construction que le notebook
    head_in = Input(shape=tuple(vgg.output.shape[1:])); x = head_in
    for layer in model.layers[model.layers.index(vgg) + 1:]:
        x = layer(x)
    head = keras.Model(head_in, x)
    grad = keras.Model(vgg.input, [vgg.get_layer("block5_conv3").output, head(vgg.output)])
    return model, grad

def _heat_demo(arr):
    import cv2
    g = arr.mean(-1); h = cv2.GaussianBlur(np.clip(g - np.percentile(g, 60), 0, None), (0, 0), 14)
    return h / (h.max() + 1e-6)

def analyse(pil, bundle):
    """Retourne (prob_pneumonie, heatmap 0-1 224x224, mode_demo)."""
    import cv2
    arr = np.array(pil.convert("RGB").resize((224, 224)), dtype="float32") / 255.0
    if bundle is None:                                    # MODE DEMO : score simule, non clinique
        c = arr[60:170, 40:184].mean()
        return float(1 / (1 + np.exp(-(c - .46) * 18))), _heat_demo(arr), True
    import tensorflow as tf
    from tensorflow.keras.applications.vgg16 import preprocess_input
    _, grad = bundle
    x = preprocess_input(arr[None] * 255.0)
    with tf.GradientTape() as tape:
        conv, pred = grad(x, training=False)
        p = float(pred[0, 0]); score = pred[:, 0] if p >= .5 else 1 - pred[:, 0]
    g = tape.gradient(score, conv)
    w = tf.reduce_mean(g, axis=(0, 1, 2))
    hm = tf.nn.relu(tf.squeeze(conv[0] @ w[..., None])).numpy()
    hm = cv2.resize(hm / (hm.max() + 1e-8), (224, 224))
    return p, hm, False

def overlay(pil, heat, alpha_max=.5):
    import cv2
    base = np.array(pil.convert("RGB").resize((224, 224)))
    col = cv2.cvtColor(cv2.applyColorMap((heat * 255).astype("uint8"), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    a = (heat[..., None] * alpha_max).astype("float32")
    return Image.fromarray((base * (1 - a) + col * a).astype("uint8")).resize((448, 448))

# ---------------------------------------------------------------- clinique
def verdict(p):
    lab = "PNEUMONIA" if p >= .5 else "NORMAL"
    conf = p if p >= .5 else 1 - p
    niveau = "ELEVEE" if conf >= .85 else "MODEREE" if conf >= .65 else "FAIBLE (zone d'incertitude)"
    return lab, conf, niveau

def recommandations(p, status="EN ATTENTE", final=None):
    lab, conf, _ = verdict(p)
    if status == "NON VALIDE" and final:
        lab = final
    r = []
    if lab == "PNEUMONIA":
        r += ["Correlation clinique immediate : fievre, toux, dyspnee, auscultation, saturation en O2.",
              "Bilan biologique : NFS, CRP, procalcitonine ; hemocultures si signes de gravite.",
              "Evaluer la gravite (score CURB-65) pour decider d'une prise en charge ambulatoire ou hospitaliere.",
              "Antibiotherapie a discretion du medecin selon les protocoles locaux et le terrain du patient.",
              "Radiographie de controle a 6-8 semaines pour verifier la resolution des opacites."]
    else:
        r += ["Aucune opacite suspecte detectee par le modele : pas d'argument radiologique pour une pneumonie.",
              "Si les symptomes persistent, envisager d'autres etiologies (bronchite, virose, cause cardiaque).",
              "Conseils : hydratation, repos, surveillance des symptomes ; reconsulter en cas d'aggravation."]
    if .35 <= p <= .65:
        r.append("Resultat en zone d'incertitude : relecture par un radiologue, voire scanner ou echographie thoracique.")
    r.append("Cet outil est une aide a la decision : la decision therapeutique appartient au medecin.")
    return r

# ---------------------------------------------------------------- dossiers
def _png(img):
    b = io.BytesIO(); img.save(b, "PNG"); return b.getvalue()

def save_case(patient, pil, heat_img, p, demo):
    cid = datetime.now().strftime("%y%m%d") + "-" + uuid.uuid4().hex[:5].upper()
    d = f"{DATA}/cases/{cid}"; os.makedirs(d)
    big = pil.convert("RGB"); big.thumbnail((900, 900)); big.save(f"{d}/original.png")
    heat_img.save(f"{d}/gradcam.png")
    lab, conf, niv = verdict(p)
    case = {"id": cid, "date": datetime.now().strftime("%d/%m/%Y %H:%M"), "patient": patient, "prob": p,
            "label": lab, "confiance": conf, "niveau": niv, "demo": demo,
            "validation": {"status": "EN ATTENTE", "medecin": "", "commentaire": "", "final": "", "date": ""}}
    json.dump(case, open(f"{d}/case.json", "w"), ensure_ascii=False, indent=1)
    return case

def list_cases():
    out = []
    for c in sorted(os.listdir(f"{DATA}/cases"), reverse=True):
        f = f"{DATA}/cases/{c}/case.json"
        if os.path.exists(f):
            out.append(json.load(open(f)))
    return out

def update_validation(case, status, medecin, commentaire, final=""):
    case["validation"] = {"status": status, "medecin": medecin, "commentaire": commentaire,
                          "final": final, "date": datetime.now().strftime("%d/%m/%Y %H:%M")}
    json.dump(case, open(f"{DATA}/cases/{case['id']}/case.json", "w"), ensure_ascii=False, indent=1)
    return case

# ---------------------------------------------------------------- PDF
def make_pdf(case):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    v, pt = case["validation"], case["patient"]
    st = getSampleStyleSheet(); H = ParagraphStyle("h", parent=st["Heading2"], textColor=colors.HexColor("#0e7490"))
    N = ParagraphStyle("n", parent=st["BodyText"], fontSize=9.5, leading=13)
    col = {"VALIDE": "#059669", "NON VALIDE": "#e11d48", "EN ATTENTE": "#d97706"}[v["status"]]
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8*cm, rightMargin=1.8*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    s = []
    head = Table([[Paragraph("<font color='white' size=18><b>RadiologyAI</b></font><br/>"
                             "<font color='#a5f3fc' size=9>Rapport d'analyse de radiographie thoracique</font>", ParagraphStyle("hd", parent=N, leading=24)),
                   Paragraph(f"<font color='white' size=9>Dossier <b>{case['id']}</b><br/>{case['date']}</font>", N)]],
                 colWidths=[12*cm, 5.4*cm])
    head.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0a1c33")),
                              ("PADDING", (0, 0), (-1, -1), 14), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    s += [head, Spacer(1, 10)]
    s.append(Paragraph("Patient", H))
    t = Table([["Nom", pt.get("nom", ""), "Age", str(pt.get("age", ""))],
               ["Sexe", pt.get("sexe", ""), "ID patient", pt.get("pid", "")],
               ["Symptomes", Paragraph(pt.get("symptomes", "") or "-", N), "", ""]],
              colWidths=[2.6*cm, 6.2*cm, 2.6*cm, 6*cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#cbd5e1")), ("FONTSIZE", (0, 0), (-1, -1), 9),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ecfeff")), ("BACKGROUND", (2, 0), (2, 1), colors.HexColor("#ecfeff")),
                           ("SPAN", (1, 2), (3, 2))]))
    s += [t, Paragraph("Imagerie et explicabilite (Grad-CAM)", H)]
    d = f"{DATA}/cases/{case['id']}"
    im = Table([[RLImage(f"{d}/original.png", 8.3*cm, 8.3*cm), RLImage(f"{d}/gradcam.png", 8.3*cm, 8.3*cm)],
                ["Radiographie originale", "Zones d'attention du modele"]], colWidths=[8.7*cm, 8.7*cm])
    im.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("FONTSIZE", (0, 1), (-1, 1), 8)]))
    s += [im, Paragraph("Resultat de l'analyse IA", H)]
    s.append(Paragraph(f"Classe predite : <b>{case['label']}</b> &nbsp; | &nbsp; Probabilite de pneumonie : <b>{case['prob']*100:.1f}%</b> "
                       f"&nbsp; | &nbsp; Confiance : {case['confiance']*100:.1f}% ({case['niveau']})", N))
    s.append(Paragraph("Validation medicale", H))
    stamp = Table([[Paragraph(f"<font color='white' size=13><b>{v['status']}</b></font>", N)]], colWidths=[5*cm])
    stamp.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(col)), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("PADDING", (0, 0), (-1, -1), 7)]))
    s += [stamp, Spacer(1, 4),
          Paragraph(f"Medecin : <b>{v['medecin'] or '-'}</b> &nbsp; Date : {v['date'] or '-'}"
                    + (f" &nbsp; Diagnostic retenu : <b>{v['final']}</b>" if v["final"] else ""), N),
          Paragraph(f"Commentaire : {v['commentaire'] or '-'}", N), Paragraph("Recommandations", H)]
    for r in recommandations(case["prob"], v["status"], v["final"]):
        s.append(Paragraph("&bull; " + r, N))
    s += [Spacer(1, 16), Paragraph("<font size=7 color='#64748b'>Document genere par RadiologyAI (VGG16, transfer learning). Aide a la decision "
                                   "non substituable au jugement clinique." + (" MODE DEMONSTRATION : score simule, non clinique." if case["demo"] else "") +
                                   "</font>", N)]
    doc.build(s)
    return buf.getvalue()
