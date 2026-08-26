-- Skema data warehouse Banking Copilot.
-- Dijalankan otomatis oleh docker-compose saat volume Postgres pertama dibuat.
--
-- Tabel gold dimaterialisasi dari parquet oleh pipelines/loaders/postgres.py
-- (to_sql if_exists='replace'), jadi berkas ini hanya menyiapkan skema, peran,
-- dan katalog kolom kunci sebagai rujukan ERD.

CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

COMMENT ON SCHEMA bronze IS 'Salinan mentah tujuh dataset publik, tanpa rekayasa nilai.';
COMMENT ON SCHEMA silver IS 'Sudah dibersihkan dan rasionya diturunkan; satu tabel per sumber.';
COMMENT ON SCHEMA gold  IS 'ERD A (inti kredit komersial) dan ERD B (lapisan graf titik-waktu).';

-- Peran baca-saja untuk aplikasi Streamlit.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'copilot_reader') THEN
        CREATE ROLE copilot_reader NOLOGIN;
    END IF;
END
$$;

GRANT USAGE ON SCHEMA gold TO copilot_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA gold GRANT SELECT ON TABLES TO copilot_reader;

-- Katalog: kolom yang tidak boleh dipakai sebagai fitur model.
CREATE TABLE IF NOT EXISTS gold.katalog_kolom_terlarang (
    tabel        text NOT NULL,
    kolom        text NOT NULL,
    alasan       text NOT NULL,
    PRIMARY KEY (tabel, kolom)
);

INSERT INTO gold.katalog_kolom_terlarang (tabel, kolom, alasan) VALUES
    ('fact_transfer_giro', 'src_is_laundering',
     'Ground truth evaluasi deteksi anomali struktural. Bukan fitur model PD.'),
    ('dim_debitur', 'label_default_debitur',
     'Target. Tidak boleh ikut ke tabel fitur graf.'),
    ('fact_laporan_keuangan', 'label_default',
     'Target PD. Dipisahkan secara fisik dari FEAT_GRAF_PIT.')
ON CONFLICT DO NOTHING;
