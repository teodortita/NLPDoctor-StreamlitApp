mkdir -p ~/.streamlit/
echo "[general]
email = \"teotita98@gmail.com\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml