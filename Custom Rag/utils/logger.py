import logging

def setup_logger(name="rag-logger", log_file="logs/app.log", level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Cegah duplikat handler jika logger dipanggil berulang
    if not logger.handlers:
        # File Handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)  # simpan semua ke file

        # Console Handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)   # hanya tampilkan INFO ke atas di terminal

        # Format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Tambahkan ke logger
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger