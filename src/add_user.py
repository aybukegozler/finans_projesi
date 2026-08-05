from database import SessionLocal, User, get_password_hash

def create_standard_user():
    db = SessionLocal()
    
    # Kullanıcı zaten var mı diye kontrol edelim
    existing_user = db.query(User).filter(User.username == "misafir_kullanici").first()
    
    if not existing_user:
        hashed_pw = get_password_hash("misafir123")
        # DİKKAT: role="user" olarak atıyoruz (admin DEĞİL)
        new_user = User(username="misafir_kullanici", hashed_password=hashed_pw, role="user")
        db.add(new_user)
        db.commit()
        print("--- Başarılı ---")
        print("Standart Kullanıcı Eklemesi Tamamlandı.")
        print("Kullanıcı Adı: misafir_kullanici | Şifre: misafir123 | Rol: USER")
    else:
        print("Bu kullanıcı zaten veritabanında mevcut.")
        
    db.close()

if __name__ == "__main__":
    create_standard_user()