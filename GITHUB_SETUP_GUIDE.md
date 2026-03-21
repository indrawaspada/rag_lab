# Panduan Menghubungkan RAG Project ke GitHub

## Step 1: Setup Git Repository Lokal

```powershell
# Buka PowerShell di folder C:\Users\indra\rag
# Inisialisasi git repository
git init

# Konfigurasi git identity
git config user.name "Indra Waspada"
git config user.email "indrawaspada@gmail.com"

# Verifikasi konfigurasi
git config --list
```

## Step 2: Buat Repository di GitHub

1. Buka https://github.com/new
2. Masukkan nama repository: `rag_lab` (atau nama lain pilihan Anda)
3. Deskripsi: "Retrieval-Augmented Generation Projects with LangChain"
4. Pilih: Public atau Private
5. **JANGAN** check "Initialize this repository with a README"
6. Klik "Create repository"

## Step 3: Hubungkan Repository Lokal ke GitHub

Setelah repository dibuat, GitHub akan menampilkan perintah. Jalankan di PowerShell:

```powershell
# Tambahkan remote repository
git remote add origin https://github.com/USERNAME/rag.git

# Ganti USERNAME dengan username GitHub Anda

# Verifikasi remote
git remote -v
```

Jika sudah ada origin dengan SSH, bisa update:
```powershell
git remote set-url origin https://github.com/USERNAME/rag.git
```

## Step 4: Commit dan Push Pertama Kali

```powershell
# Tambahkan semua file (sudah mengikuti .gitignore)
git add .

# Cek status
git status

# Commit
git commit -m "Initial commit: RAG projects with LangChain and OpenAI"

# Rename branch ke main (jika belum)
git branch -M main

# Push ke GitHub
git push -u origin main
```

## Step 5: Setelah Push Pertama

Untuk push selanjutnya, cukup:
```powershell
git add .
git commit -m "Your message here"
git push
```

## Workflow Standar Git

### Membuat Fitur Baru
```powershell
# Buat branch baru
git checkout -b feature/nama-fitur

# Lakukan perubahan, commit
git add .
git commit -m "Add new feature: deskripsi fitur"

# Push branch
git push -u origin feature/nama-fitur

# Buat Pull Request di GitHub
# Merge ke main setelah review
```

### Update dari GitHub
```powershell
git pull origin main
```

## Tips Penting

✅ **Commit messages yang baik:**
- `git commit -m "Add CSV RAG implementation"` ✓ Good
- `git commit -m "fix"` ✗ Bad

✅ **Jangan commit:**
- `.env` file (API keys) - sudah di .gitignore
- `venv/` atau `ragenv/` folder
- `__pycache__/`
- `.ipynb_checkpoints/`

✅ **File yang harus ada di repo:**
- `README.md` atau `README_GITHUB.md`
- `Requirements.txt` (sudah ada)
- `.gitignore` (sudah ada)
- Source code files
- Documentation

## Troubleshooting

### Error: "fatal: remote origin already exists"
```powershell
git remote remove origin
git remote add origin https://github.com/USERNAME/rag.git
```

### Error: "Permission denied (publickey)"
Gunakan HTTPS URL daripada SSH:
```powershell
git remote set-url origin https://github.com/USERNAME/rag.git
```

### Error: "Updates were rejected"
```powershell
# Pull changes terlebih dahulu
git pull origin main

# Jika ada conflict, resolve kemudian commit
git add .
git commit -m "Merge changes"
git push origin main
```

## Menggunakan GitHub Desktop (Alternative)

Jika prefer GUI:
1. Download GitHub Desktop dari https://desktop.github.com/
2. Sign in dengan GitHub account
3. File → Clone Repository → masukkan URL
4. Mulai commit dan push dari aplikasi

## Next Steps

Setelah berhasil connect:
1. Update README.md dengan deskripsi project
2. Add documentation di folder `docs/`
3. Buat issues untuk tracking features/bugs
4. Setup GitHub Actions untuk CI/CD (optional)

---

**Perlu bantuan?** 
- Dokumentasi GitHub: https://docs.github.com
- GitHub Learning Lab: https://github.blog/2020-04-07-github-cli-is-now-available/
