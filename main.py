import json
from collections import Counter
import matplotlib.pyplot as plt

# Datei einlesen
def load_data(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        return json.load(file)

# Analyse starten
def analyze_data(data):
    total_domains = len(data)
    
    # Initialisierung von Statistiken
    missing_security_headers = Counter()
    open_ports_counter = Counter()
    domains_with_hsts = 0
    domains_with_tls = 0
    valid_ssl_certificates = 0
    expired_ssl_certificates = 0
    dnssec_issues = Counter()

    for domain in data:
        domain_name = domain.get("domain", "Unknown")
        endpoints = domain.get("endpoints", [])

        for endpoint in endpoints:
            response = endpoint.get("response", {})
            
            # Sicherheitsheader prüfen
            if endpoint["endpoint"] == "/http-headers":
                security_headers = response.get("securityHeaders", {})
                for header, present in security_headers.items():
                    if not present:
                        missing_security_headers[header] += 1

            # Geöffnete Ports sammeln
            if endpoint["endpoint"] == "/ports":
                open_ports = response.get("openPorts", [])
                open_ports_counter.update(open_ports)

            # HSTS prüfen
            if endpoint["endpoint"] == "/hsts":
                if response.get("hsts", False):
                    domains_with_hsts += 1

            # TLS-Prüfung
            if endpoint["endpoint"] == "/tls":
                if response.get("tls", {}):
                    domains_with_tls += 1

            # SSL-Zertifikatsprüfung
            if endpoint["endpoint"] == "/ssl":
                ssl_info = response.get("ssl", {})
                valid_to = ssl_info.get("valid_to")
                if valid_to:
                    valid_ssl_certificates += 1

            # DNSSEC-Fehler prüfen
            if endpoint["endpoint"] == "/dnssec":
                for key in ["DNSKEY", "DS", "RRSIG"]:
                    dnssec_info = response.get(key, {})
                    if not dnssec_info.get("isFound", False):
                        dnssec_issues[key] += 1

    # Ergebnisse ausgeben
    print(f"Gesamtanzahl der Domains: {total_domains}")
    print("\nFehlende Sicherheitsheader:")
    for header, count in missing_security_headers.items():
        print(f"  {header}: {count}")

    print("\nHäufigkeit geöffneter Ports:")
    for port, count in open_ports_counter.most_common():
        print(f"  Port {port}: {count}")

    print(f"\nDomains mit HSTS aktiviert: {domains_with_hsts}")
    print(f"Domains mit TLS aktiviert: {domains_with_tls}")
    print(f"Gültige SSL-Zertifikate: {valid_ssl_certificates}")
    print("\nDNSSEC-Fehler:")
    for issue, count in dnssec_issues.items():
        print(f"  {issue}: {count}")

    # Visuelle Darstellung der Ergebnisse
    visualize_results(missing_security_headers, open_ports_counter, domains_with_hsts, domains_with_tls, valid_ssl_certificates, dnssec_issues)

# Visuelle Darstellung
def visualize_results(missing_security_headers, open_ports_counter, domains_with_hsts, domains_with_tls, valid_ssl_certificates, dnssec_issues):
    # Fehlende Sicherheitsheader
    plt.figure(figsize=(10, 6))
    headers, counts = zip(*missing_security_headers.items())
    plt.bar(headers, counts, color='orange')
    plt.title('Fehlende Sicherheitsheader')
    plt.xlabel('Header')
    plt.ylabel('Anzahl der Domains')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Häufigkeit geöffneter Ports
    plt.figure(figsize=(10, 6))
    ports, counts = zip(*open_ports_counter.most_common())
    plt.bar(ports, counts, color='blue')
    plt.title('Häufigkeit geöffneter Ports')
    plt.xlabel('Port')
    plt.ylabel('Anzahl der Domains')
    plt.tight_layout()
    plt.show()

    # HSTS und TLS
    labels = ['HSTS aktiviert', 'TLS aktiviert', 'Gültige SSL-Zertifikate']
    values = [domains_with_hsts, domains_with_tls, valid_ssl_certificates]
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['green', 'purple', 'cyan'])
    plt.title('HSTS, TLS und SSL-Zertifikate')
    plt.ylabel('Anzahl der Domains')
    plt.tight_layout()
    plt.show()

    # DNSSEC-Fehler
    plt.figure(figsize=(10, 6))
    issues, counts = zip(*dnssec_issues.items())
    plt.bar(issues, counts, color='red')
    plt.title('DNSSEC-Fehler')
    plt.xlabel('Fehler-Typ')
    plt.ylabel('Anzahl der Domains')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_name = "StatisticsMittelschulen.json"
    data = load_data(file_name)
    analyze_data(data)
