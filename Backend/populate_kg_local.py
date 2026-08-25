"""
populate_kg_local.py
====================
Script untuk menambahkan data Knowledge Graph dari notebook 
(integrasi-neo4j-aura.ipynb) ke Neo4j lokal TANPA menghapus 
data lama dan TANPA menduplikasi node yang sudah ada.

Menggunakan MERGE sehingga aman dijalankan berulang kali.
"""

from neo4j import GraphDatabase
import sys

# Ensure UTF-8 output to prevent UnicodeEncodeError in Windows CMD/PowerShell
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import os

# ── Konfigurasi Neo4j Lokal ──────────────────────────────────
URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
DATABASE = os.getenv("NEO4J_DATABASE", None)

# ── Data 10 Spesies Hoya ─────────────────────────────────────
species_list = [
    "Hoya Macrophylla", "Hoya Wayeti", "Hoya australis",
    "Hoya callistophylla", "Hoya finlaysonii", "Hoya imperialis",
    "Hoya kerrii", "Hoya lacunosa", "Hoya pubicalyx", "hoya carnosa"
]

# ── Kategori Pemanfaatan Tiap Spesies ────────────────────────
species_categories = {
    "Hoya Macrophylla": ["Tanaman Hias"],
    "Hoya Wayeti": ["Tanaman Hias"],
    "Hoya australis": ["Tanaman Hias", "Tanaman Obat"],
    "Hoya callistophylla": ["Tanaman Hias"],
    "Hoya finlaysonii": ["Tanaman Hias", "Tanaman Obat"],
    "Hoya imperialis": ["Tanaman Hias"],
    "Hoya kerrii": ["Tanaman Hias", "Tanaman Obat"],
    "Hoya lacunosa": ["Tanaman Hias", "Tanaman Obat"],
    "Hoya pubicalyx": ["Tanaman Hias"],
    "hoya carnosa": ["Tanaman Hias", "Tanaman Obat"],
}

# ── Data 9 Penyakit ──────────────────────────────────────────
diseases = [
    {"en": "Botrytis Blight", "id": "Busuk Abu-abu", "category": "Bercak Cokelat"},
    {"en": "Anthracnose / Leaf Spot Disease", "id": "Antraknosa / Penyakit Bercak Daun", "category": "Bercak Cokelat"},
    {"en": "Bacterial Blight", "id": "Hawar Bakteri", "category": "Bercak Cokelat"},
    {"en": "Powdery Mildew", "id": "Embun Tepung", "category": "Bercak Putih"},
    {"en": "Sooty Mold", "id": "Embun Jelaga", "category": "Bercak Bintik Hitam"},
    {"en": "Unspecified Fungal/Bacterial Leaf Spot", "id": "Bercak Daun Jamur/Bakteri Tidak Spesifik", "category": "Bercak Bintik Hitam"},
    {"en": "Root Rot", "id": "Busuk Akar", "category": "Daun Layu"},
    {"en": "Cercospora", "id": "Bercak Mata Katak", "category": "Bercak Cokelat"},
    {"en": "Rust", "id": "Karat Daun", "category": "Bercak Cokelat"}
]

# ── Data 3 Hama ──────────────────────────────────────────────
pests = [
    {"en": "Mealybugs", "id": "Kutu Putih", "category": "Bercak Putih"},
    {"en": "Aphids", "id": "Kutu Daun", "category": "Bercak Bintik Hitam"},
    {"en": "Spider Mites", "id": "Tungau Laba-laba", "category": None}
]

# ── Struktur Pengetahuan Relasional Bilingual ────────────────
knowledge = {
    "Botrytis Blight": {
        "symptoms": [
            {"id": "Bercak abu-abu/cokelat berair pada daun", "en": "Water-soaked gray/brown spots on leaves"},
            {"id": "Daun lunak dan membusuk", "en": "Leaves become soft and rot"},
            {"id": "Muncul jamur putih/abu-abu/cokelat gelap", "en": "Appearance of white/gray/dark brown mold"},
            {"id": "Batang dan bunga ikut layu", "en": "Stems and flowers wilt"}
        ],
        "causes": [
            {"id": "Kelembapan tinggi", "en": "High humidity"},
            {"id": "Suhu dingin", "en": "Cold temperatures"},
            {"id": "Sirkulasi udara buruk", "en": "Poor air circulation"},
            {"id": "Sisa bunga/daun mati yang menempel di tanaman", "en": "Dead flowers/leaves clinging to the plant"}
        ],
        "treatments": [
            {"id": "Isolasi tanaman", "en": "Isolate the plant"},
            {"id": "Buang seluruh bagian terinfeksi", "en": "Remove all infected parts"},
            {"id": "Tingkatkan sirkulasi udara", "en": "Improve air circulation"},
            {"id": "Aplikasikan fungisida (umum)", "en": "Apply general fungicide"}
        ]
    },
    "Anthracnose / Leaf Spot Disease": {
        "symptoms": [
            {"id": "Bercak kecil yang membesar dan bertambah banyak", "en": "Small spots that enlarge and multiply"},
            {"id": "Berkembang dari kuning menjadi cokelat hingga lesi cekung gelap", "en": "Develops from yellow to brown to dark sunken lesions"},
            {"id": "Kadang disertai halo kuning", "en": "Sometimes accompanied by a yellow halo"}
        ],
        "causes": [
            {"id": "Air menggenang di daun", "en": "Water pooling on leaves"},
            {"id": "Percikan tanah saat penyiraman", "en": "Soil splashing during watering"},
            {"id": "Alat pangkas terkontaminasi", "en": "Contaminated pruning tools"},
            {"id": "Sirkulasi udara buruk", "en": "Poor air circulation"}
        ],
        "treatments": [
            {"id": "Buang daun terinfeksi", "en": "Remove infected leaves"},
            {"id": "Siram ke media, bukan ke daun", "en": "Water the soil, not the leaves"},
            {"id": "Tingkatkan sirkulasi udara", "en": "Improve air circulation"},
            {"id": "Aplikasikan fungisida tembaga", "en": "Apply copper fungicide"}
        ]
    },
    "Bacterial Blight": {
        "symptoms": [
            {"id": "Bercak kecil berair (water-soaked spots)", "en": "Small water-soaked spots"},
            {"id": "Membesar menjadi area cokelat nekrotik", "en": "Enlarges into brown necrotic areas"},
            {"id": "Muncul cairan lengket (ooze) pada kondisi lembap", "en": "Sticky ooze appears in humid conditions"}
        ],
        "causes": [
            {"id": "Kelembapan tinggi", "en": "High humidity"},
            {"id": "Penyiraman dari atas", "en": "Overhead watering"},
            {"id": "Luka pada jaringan daun", "en": "Wounds on leaf tissue"}
        ],
        "treatments": [
            {"id": "Isolasi tanaman", "en": "Isolate the plant"},
            {"id": "Pangkas dengan alat steril", "en": "Prune with sterile tools"},
            {"id": "Hindari penyiraman dari atas", "en": "Avoid overhead watering"},
            {"id": "Perbaiki pencahayaan dan sirkulasi udara", "en": "Improve lighting and air circulation"}
        ]
    },
    "Powdery Mildew": {
        "symptoms": [
            {"id": "Bercak putih seperti tepung", "en": "White powdery spots"},
            {"id": "Meluas menutupi daun dan batang", "en": "Spreads to cover leaves and stems"},
            {"id": "Daun menggulung", "en": "Curling leaves"},
            {"id": "Pertumbuhan melambat", "en": "Stunted growth"}
        ],
        "causes": [
            {"id": "Sirkulasi udara buruk", "en": "Poor air circulation"},
            {"id": "Tanaman terlalu rapat", "en": "Overcrowded plants"},
            {"id": "Ruang tertutup", "en": "Enclosed space"}
        ],
        "treatments": [
            {"id": "Isolasi tanaman", "en": "Isolate the plant"},
            {"id": "Buang daun terserang berat", "en": "Remove heavily infected leaves"},
            {"id": "Tingkatkan sirkulasi udara", "en": "Improve air circulation"},
            {"id": "Aplikasikan fungisida khusus", "en": "Apply specific fungicide"}
        ]
    },
    "Sooty Mold": {
        "symptoms": [
            {"id": "Lapisan hitam seperti jelaga pada daun dan batang", "en": "Soot-like black layer on leaves and stems"},
            {"id": "Mudah terhapus dengan kain lembap", "en": "Easily wiped off with a damp cloth"},
            {"id": "Residu lengket sebelum jelaga muncul", "en": "Sticky residue before mold appears"}
        ],
        "causes": [
            {"id": "Hama pengisap getah penghasil honeydew", "en": "Sap-sucking pests producing honeydew"}
        ],
        "treatments": [
            {"id": "Kendalikan hama (sabun insektisida/neem oil)", "en": "Control pests (insecticidal soap/neem oil)"},
            {"id": "Bersihkan jelaga dengan kain lembap", "en": "Clean mold with a damp cloth"}
        ]
    },
    "Unspecified Fungal/Bacterial Leaf Spot": {
        "symptoms": [
            {"id": "Bercak cokelat/hitam bulat", "en": "Round brown/black spots"},
            {"id": "Kadang disertai halo kuning", "en": "Sometimes accompanied by a yellow halo"}
        ],
        "causes": [
            {"id": "Daun basah berkepanjangan", "en": "Prolonged leaf wetness"},
            {"id": "Sirkulasi udara buruk", "en": "Poor air circulation"},
            {"id": "Alat pangkas terkontaminasi", "en": "Contaminated pruning tools"}
        ],
        "treatments": [
            {"id": "Buang daun terinfeksi", "en": "Remove infected leaves"},
            {"id": "Tingkatkan sirkulasi udara", "en": "Improve air circulation"},
            {"id": "Aplikasikan fungisida tembaga", "en": "Apply copper fungicide"}
        ]
    },
    "Root Rot": {
        "symptoms": [
            {"id": "Daun menguning dan layu, terasa lunak", "en": "Yellowing and wilting leaves, feeling soft"},
            {"id": "Batang lembek dekat media tanam", "en": "Soft stems near the potting mix"},
            {"id": "Akar berubah abu-abu/cokelat/hitam dan lembek", "en": "Roots turn gray/brown/black and soft"},
            {"id": "Bau apek/busuk dari media tanam", "en": "Musty/foul odor from the potting mix"}
        ],
        "causes": [
            {"id": "Penyiraman berlebih dalam waktu lama", "en": "Prolonged overwatering"},
            {"id": "Media tanam padat atau drainase buruk", "en": "Compacted potting mix or poor drainage"}
        ],
        "treatments": [
            {"id": "Keluarkan tanaman dari pot", "en": "Remove the plant from its pot"},
            {"id": "Potong akar yang busuk", "en": "Trim rotted roots"},
            {"id": "Tanam ulang di media porous dengan drainase baik", "en": "Repot in porous mix with good drainage"},
            {"id": "Siram hanya saat media benar-benar kering", "en": "Water only when the mix is completely dry"}
        ]
    },
    "Cercospora": {
        "symptoms": [
            {"id": "Terdapat lesi atau bintik kecil berwarna abu-abu pada bagian tengah", "en": "Gray lesions or small spots in the center"},
            {"id": "Tepi bintik berwarna cokelat kemerahan atau ungu", "en": "Reddish-brown or purple edges around spots"}
        ],
        "causes": [
            {"id": "Kelembapan berlebihan di udara", "en": "Excessive air humidity"},
            {"id": "Sirkulasi udara buruk", "en": "Poor air circulation"}
        ],
        "treatments": [
            {"id": "Isolasi tanaman dan potong daun sakit", "en": "Isolate plant and cut sick leaves"},
            {"id": "Semprot dengan fungisida tembaga", "en": "Spray with copper fungicide"},
            {"id": "Hindari membasahi daun saat menyiram", "en": "Avoid wetting leaves during watering"}
        ]
    },
    "Rust": {
        "symptoms": [
            {"id": "Terdapat pustul atau bintik menonjol di daun", "en": "Pustules or raised spots on leaves"},
            {"id": "Berwarna oranye, merah karat, atau cokelat", "en": "Orange, rust-red, or brown color"}
        ],
        "causes": [
            {"id": "Suhu hangat dengan kelembapan tinggi", "en": "Warm temperature with high humidity"},
            {"id": "Terpapar tanaman terinfeksi jamur karat", "en": "Exposure to plants infected with rust fungi"}
        ],
        "treatments": [
            {"id": "Segera buang daun yang terinfeksi", "en": "Immediately discard infected leaves"},
            {"id": "Jaga daun tetap kering", "en": "Keep leaves dry"},
            {"id": "Beri fungisida berbahan aktif sulfur atau tembaga", "en": "Apply sulfur or copper-based fungicide"}
        ]
    },
    "Mealybugs": {
        "symptoms": [
            {"id": "Bercak putih seperti kapas di sela daun atau batang", "en": "White cotton-like spots in leaf axils or stems"},
            {"id": "Daun terlihat kusam dan layu", "en": "Leaves look dull and wilted"},
            {"id": "Terasa lengket (embun madu) dan sering mengundang semut", "en": "Sticky feeling (honeydew) and attracts ants"}
        ],
        "causes": [
            {"id": "Lingkungan kering dan hangat", "en": "Dry and warm environment"},
            {"id": "Tertular dari tanaman baru", "en": "Contagion from new plants"}
        ],
        "treatments": [
            {"id": "Bersihkan hama dengan kapas beralkohol (70%)", "en": "Clean pests with 70% alcohol cotton swabs"},
            {"id": "Gunakan sabun insektisida atau neem oil", "en": "Use insecticidal soap or neem oil"},
            {"id": "Rutin periksa ketiak daun", "en": "Regularly inspect leaf axils"}
        ]
    },
    "Aphids": {
        "symptoms": [
            {"id": "Serangga kecil bergerombol di pucuk daun muda", "en": "Small insects clustering on new shoots"},
            {"id": "Daun mengeriting dan berubah bentuk", "en": "Curling and deformed leaves"},
            {"id": "Muncul jamur jelaga hitam akibat honeydew", "en": "Black sooty mold appears due to honeydew"},
            {"id": "Pertumbuhan tanaman terhambat", "en": "Stunted plant growth"}
        ],
        "causes": [
            {"id": "Perkembangbiakan hama sangat cepat", "en": "Very rapid pest multiplication"},
            {"id": "Terbawa angin atau semut", "en": "Carried by wind or ants"}
        ],
        "treatments": [
            {"id": "Semprot dengan aliran air kuat untuk menjatuhkan hama", "en": "Spray with strong water jet to dislodge pests"},
            {"id": "Semprotkan minyak mimba (neem oil)", "en": "Spray neem oil"},
            {"id": "Beri pupuk berimbang agar tanaman tahan stres", "en": "Apply balanced fertilizer for stress resistance"}
        ]
    },
    "Spider Mites": {
        "symptoms": [
            {"id": "Daun memiliki bintik-bintik kuning kecil dan kusam", "en": "Leaves have small dull yellow spots"},
            {"id": "Terlihat jaring halus di sekitar pangkal daun", "en": "Fine webbing visible near leaf bases"},
            {"id": "Daun menguning dan rontok prematur", "en": "Premature yellowing and leaf drop"}
        ],
        "causes": [
            {"id": "Udara ruangan sangat panas dan kering", "en": "Very hot and dry indoor air"},
            {"id": "Kurang penyiraman", "en": "Underwatering"}
        ],
        "treatments": [
            {"id": "Lap daun dengan kain lembap", "en": "Wipe leaves with a damp cloth"},
            {"id": "Tingkatkan kelembapan sekitar tanaman", "en": "Increase humidity around the plant"},
            {"id": "Gunakan mitisida jika serangan parah", "en": "Use miticide if infestation is severe"}
        ]
    }
}


def main():
    print("=" * 60)
    print("  Populate Knowledge Graph ke Neo4j Lokal")
    print("  (Mode: MERGE — data lama TIDAK dihapus)")
    print("=" * 60)

    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

    # Tes koneksi
    try:
        driver.verify_connectivity()
        print("\n✅ Berhasil terhubung ke Neo4j!\n")
    except Exception as e:
        print(f"\n❌ Gagal terhubung ke Neo4j: {e}")
        print("   Pastikan Neo4j Desktop sedang berjalan.")
        return

    session_kwargs = {'database': DATABASE} if DATABASE else {}
    with driver.session(**session_kwargs) as session:

        # ── 1. Buat Constraints (IF NOT EXISTS = aman) ────────
        print("📌 Membuat constraints (jika belum ada)...")
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:HoyaSpecies) REQUIRE s.name_en IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.name_en IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (pst:Pest) REQUIRE pst.name_en IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (sym:Symptom) REQUIRE sym.name_en IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:CausalFactor) REQUIRE c.name_en IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Treatment) REQUIRE t.name_en IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:PlantCategory) REQUIRE p.name IS UNIQUE"
        ]
        for q in constraints:
            session.run(q)
        print("   ✅ Constraints siap.\n")

        # ── 2. MERGE Spesies (10 Hoya) ────────────────────────
        print("🌱 Menambahkan 10 Spesies Hoya...")
        for sp in species_list:
            session.run(
                "MERGE (s:HoyaSpecies {name_en: $name}) "
                "SET s.name_id = $name",
                name=sp
            )
        print(f"   ✅ {len(species_list)} spesies diproses (MERGE).\n")

        # ── 3. MERGE Kategori Tanaman + Relasi ────────────────
        print("🏷️  Menambahkan Kategori Tanaman & Relasi...")
        cat_count = 0
        for sp, categories in species_categories.items():
            for cat in categories:
                session.run(
                    """
                    MATCH (s:HoyaSpecies {name_en: $sp_name})
                    MERGE (p:PlantCategory {name: $cat_name})
                    MERGE (s)-[:TERGOLONG_SEBAGAI]->(p)
                    """,
                    sp_name=sp, cat_name=cat
                )
                cat_count += 1
        print(f"   ✅ {cat_count} relasi kategori diproses (MERGE).\n")

        # ── 4. MERGE Penyakit (9 jenis) ───────────────────────
        print("🦠 Menambahkan 9 Penyakit...")
        for d in diseases:
            session.run(
                "MERGE (dis:Disease {name_en: $en}) "
                "SET dis.name_id = $id, dis.category = $cat",
                en=d["en"], id=d["id"], cat=d["category"]
            )
        print(f"   ✅ {len(diseases)} penyakit diproses (MERGE).\n")

        # ── 4.5. MERGE Hama (3 jenis) ───────────────────────
        print("🐛 Menambahkan 3 Hama...")
        for p in pests:
            session.run(
                "MERGE (pst:Pest {name_en: $en}) "
                "SET pst.name_id = $id, pst.category = $cat",
                en=p["en"], id=p["id"], cat=p["category"]
            )
        print(f"   ✅ {len(pests)} hama diproses (MERGE).\n")

        # ── 5. MERGE Gejala, Penyebab, Penanganan + Relasi ───
        print("🔗 Menambahkan Gejala, Penyebab, Penanganan beserta Relasi...")
        sym_count, cause_count, treat_count = 0, 0, 0

        for entity_en, data in knowledge.items():

            label = "Disease" if any(d["en"] == entity_en for d in diseases) else "Pest" if any(p["en"] == entity_en for p in pests) else None
            if not label:
                continue

            # Gejala (HAS_SYMPTOM) - diset properti category agar query diawali dari Node Symptom
            cat = next((d["category"] for d in diseases if d["en"] == entity_en), None)
            if not cat:
                cat = next((p["category"] for p in pests if p["en"] == entity_en), None)

            for sym in data["symptoms"]:
                session.run(
                    f"""
                    MATCH (n:{label} {{name_en: $name}})
                    MERGE (s:Symptom {{name_en: $s_en}})
                    SET s.name_id = $s_id, s.category = $cat
                    MERGE (n)-[:HAS_SYMPTOM]->(s)
                    """,
                    name=entity_en, s_en=sym["en"], s_id=sym["id"], cat=cat
                )
                sym_count += 1

            # Penyebab (FAVORED_BY)
            for cause in data["causes"]:
                session.run(
                    f"""
                    MATCH (n:{label} {{name_en: $name}})
                    MERGE (c:CausalFactor {{name_en: $c_en}})
                    SET c.name_id = $c_id
                    MERGE (n)-[:FAVORED_BY]->(c)
                    """,
                    name=entity_en, c_en=cause["en"], c_id=cause["id"]
                )
                cause_count += 1

            # Penanganan (TREATED_WITH)
            for treatment in data["treatments"]:
                session.run(
                    f"""
                    MATCH (n:{label} {{name_en: $name}})
                    MERGE (t:Treatment {{name_en: $t_en}})
                    SET t.name_id = $t_id
                    MERGE (n)-[:TREATED_WITH]->(t)
                    """,
                    name=entity_en, t_en=treatment["en"], t_id=treatment["id"]
                )
                treat_count += 1

        print(f"   ✅ {sym_count} gejala diproses.")
        print(f"   ✅ {cause_count} penyebab diproses.")
        print(f"   ✅ {treat_count} penanganan diproses.")

    driver.close()

    print("\n" + "=" * 60)
    print("  🎉 SELESAI! Knowledge Graph berhasil diperbarui.")
    print("  Data lama tetap utuh, data baru sudah ditambahkan.")
    print("=" * 60)


if __name__ == "__main__":
    main()
