import json
from collections import Counter
import matplotlib.pyplot as plt
from flask import Flask, render_template_string
import io
import base64
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage


# Flask-Setup
app = Flask(__name__)

# HTML-Vorlage mit Tailwind CSS
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Statistik der Domains</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 text-gray-800">
    <div class="container mx-auto px-4 py-6">
        <h1 class="text-3xl font-bold mb-6 text-center text-blue-600">Statistische Auswertung der Domains</h1>
        
        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-2xl font-semibold text-gray-700 mb-4">Fehlende Sicherheitsheader</h2>
            <img src="data:image/png;base64,{{ missing_headers_plot }}" alt="Fehlende Sicherheitsheader" class="rounded-lg shadow-md mx-auto">
        </div>

        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-2xl font-semibold text-gray-700 mb-4">Häufigkeit geöffneter Ports</h2>
            <img src="data:image/png;base64,{{ open_ports_plot }}" alt="Häufigkeit geöffneter Ports" class="rounded-lg shadow-md mx-auto">
        </div>

        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-2xl font-semibold text-gray-700 mb-4">HSTS, TLS und SSL-Zertifikate</h2>
            <img src="data:image/png;base64,{{ hsts_tls_plot }}" alt="HSTS, TLS und SSL-Zertifikate" class="rounded-lg shadow-md mx-auto">
        </div>

        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-2xl font-semibold text-gray-700 mb-4">DNSSEC-Fehler</h2>
            <img src="data:image/png;base64,{{ dnssec_issues_plot }}" alt="DNSSEC-Fehler" class="rounded-lg shadow-md mx-auto">
        </div>

        <!-- Clusteranalyse -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-2xl font-semibold text-gray-700 mb-4">Clusteranalyse der Domains</h2>
            <img src="data:image/png;base64,{{ cluster_plot }}" alt="Clusteranalyse" class="rounded-lg shadow-md mx-auto">
        </div>
        
        <!-- Hierarchisches Clustern -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-2xl font-semibold text-gray-700 mb-4">Hierarchische Clusteranalyse der Domains</h2>
            <img src="data:image/png;base64,{{ hierarchical_cluster_plot }}" alt="Hierarchische Clusteranalyse" class="rounded-lg shadow-md mx-auto">
        </div>

    </div>
</body>
</html>
"""

# Datei einlesen
def load_data(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        return json.load(file)

def hierarchical_cluster_analysis(data):
    features = []
    for domain in data:
        endpoints = domain.get("endpoints", [])
        open_ports = 0
        security_headers_missing = 0
        hsts = 0
        tls = 0

        for endpoint in endpoints:
            response = endpoint.get("response", {})
            if endpoint["endpoint"] == "/ports":
                open_ports = len(response.get("openPorts", []))
            if endpoint["endpoint"] == "/http-headers":
                security_headers_missing = sum(1 for _, present in response.get("securityHeaders", {}).items() if not present)
            if endpoint["endpoint"] == "/hsts":
                hsts = 1 if response.get("hsts", False) else 0
            if endpoint["endpoint"] == "/tls":
                tls = 1 if response.get("tls", False) else 0

        features.append([open_ports, security_headers_missing, hsts, tls])

    # Normalisiere die Daten
    features = np.array(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Hierarchisches Clustering (Linkage-Methode)
    linked = linkage(features_scaled, method='ward')

    # Dendrogramm erstellen
    plt.figure(figsize=(10, 6))
    dendrogram(linked)
    plt.title("Hierarchisches Clustering von Domains")
    plt.xlabel("Domain-Indizes")
    plt.ylabel("Abstand")
    plt.tight_layout()
    return create_plot()


def cluster_analysis(data):
    features = []
    for domain in data:
        endpoints = domain.get("endpoints", [])
        open_ports = 0
        security_headers_missing = 0
        hsts = 0
        tls = 0

        for endpoint in endpoints:
            response = endpoint.get("response", {})
            if endpoint["endpoint"] == "/ports":
                open_ports = len(response.get("openPorts", []))
            if endpoint["endpoint"] == "/http-headers":
                security_headers_missing = sum(1 for _, present in response.get("securityHeaders", {}).items() if not present)
            if endpoint["endpoint"] == "/hsts":
                hsts = 1 if response.get("hsts", False) else 0
            if endpoint["endpoint"] == "/tls":
                tls = 1 if response.get("tls", False) else 0

        features.append([open_ports, security_headers_missing, hsts, tls])

    # Normalisiere die Daten
    features = np.array(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)

    # Scatterplot erstellen
    plt.figure(figsize=(10, 6))
    plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=clusters, cmap="viridis", s=50)
    plt.title("Clusteranalyse von Domains")
    plt.xlabel("Geöffnete Ports (skaliert)")
    plt.ylabel("Fehlende Sicherheitsheader (skaliert)")
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    return create_plot()

# Diagramm erstellen und als Base64 zurückgeben
def create_plot():
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close()
    return img_base64

# Analyse starten
@app.route("/")
def analyze_data():
    data = load_data("StatisticsMittelschulen.json")
    total_domains = len(data)
    cluster_plot = cluster_analysis(data)
        # Hierarchische Clusteranalyse
    hierarchical_cluster_plot = hierarchical_cluster_analysis(data)


    
    missing_security_headers = Counter()
    open_ports_counter = Counter()
    domains_with_hsts = 0
    domains_with_tls = 0
    valid_ssl_certificates = 0
    dnssec_issues = Counter()

    for domain in data:
        domain_name = domain.get("domain", "Unknown")
        endpoints = domain.get("endpoints", [])

        for endpoint in endpoints:
            response = endpoint.get("response", {})
            
            if endpoint["endpoint"] == "/http-headers":
                security_headers = response.get("securityHeaders", {})
                for header, present in security_headers.items():
                    if not present:
                        missing_security_headers[header] += 1

            if endpoint["endpoint"] == "/ports":
                open_ports = response.get("openPorts", [])
                open_ports_counter.update(open_ports)

            if endpoint["endpoint"] == "/hsts":
                if response.get("hsts", False):
                    domains_with_hsts += 1

            if endpoint["endpoint"] == "/tls":
                if response.get("tls", {}):
                    domains_with_tls += 1

            if endpoint["endpoint"] == "/ssl":
                ssl_info = response.get("ssl", {})
                valid_to = ssl_info.get("valid_to")
                if valid_to:
                    valid_ssl_certificates += 1

            if endpoint["endpoint"] == "/dnssec":
                for key in ["DNSKEY", "DS", "RRSIG"]:
                    dnssec_info = response.get(key, {})
                    if not dnssec_info.get("isFound", False):
                        dnssec_issues[key] += 1

    # Fehlende Sicherheitsheader
    plt.figure(figsize=(10, 6))
    headers, counts = zip(*missing_security_headers.items())
    plt.bar(headers, counts, color='orange')
    plt.title('Fehlende Sicherheitsheader')
    plt.xlabel('Header')
    plt.ylabel('Anzahl der Domains')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    missing_headers_plot = create_plot()

    # Häufigkeit geöffneter Ports
    plt.figure(figsize=(10, 6))
    ports, counts = zip(*open_ports_counter.most_common())
    ports = [str(port) for port in ports]
    plt.bar(ports, counts, color='blue')
    plt.title('Häufigkeit geöffneter Ports')
    plt.xlabel('Port')
    plt.ylabel('Anzahl der Domains')
    plt.tight_layout()
    open_ports_plot = create_plot()

    # HSTS, TLS und SSL-Zertifikate
    labels = ['HSTS aktiviert', 'TLS aktiviert', 'Gültige SSL-Zertifikate']
    values = [domains_with_hsts, domains_with_tls, valid_ssl_certificates]
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['green', 'purple', 'cyan'])
    plt.title('HSTS, TLS und SSL-Zertifikate')
    plt.ylabel('Anzahl der Domains')
    plt.tight_layout()
    hsts_tls_plot = create_plot()

    # DNSSEC-Fehler
    plt.figure(figsize=(10, 6))
    issues, counts = zip(*dnssec_issues.items())
    plt.bar(issues, counts, color='red')
    plt.title('DNSSEC-Fehler')
    plt.xlabel('Fehler-Typ')
    plt.ylabel('Anzahl der Domains')
    plt.tight_layout()
    dnssec_issues_plot = create_plot()

    return render_template_string(html_template,
                                  missing_headers_plot=missing_headers_plot,
                                  open_ports_plot=open_ports_plot,
                                  hsts_tls_plot=hsts_tls_plot,
                                  dnssec_issues_plot=dnssec_issues_plot,  cluster_plot=cluster_plot, hierarchical_cluster_plot=hierarchical_cluster_plot)

if __name__ == "__main__":
    app.run(debug=True)
