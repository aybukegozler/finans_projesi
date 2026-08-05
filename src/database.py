from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
import bcrypt
import os

# Veritabanı dosyasının data klasörü içinde oluşturulması
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DB_DIR, exist_ok=True)

SQLALCHEMY_DATABASE_URL = f"sqlite:///{os.path.join(DB_DIR, 'quant_app.db')}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Kullanıcı (User) Tablosunun Şeması
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="user") # 'admin' veya 'user'

# Veritabanı tablolarını oluştur
Base.metadata.create_all(bind=engine)

# Passlib yerine doğrudan native bcrypt fonksiyonu kullanıyoruz
def get_password_hash(password: str):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

# İlk kurulumda test edebilmemiz için varsayılan bir yönetici (admin) hesabı ekleyelim
def create_initial_admin():
    db = SessionLocal()
    admin = db.query(User).filter(User.username == "aybuke_admin").first()
    if not admin:
        hashed_pw = get_password_hash("admin123") 
        new_admin = User(username="aybuke_admin", hashed_password=hashed_pw, role="admin")
        db.add(new_admin)
        db.commit()
        print("--- Veritabanı Başarıyla Kuruldu ---")
        print("Yönetici Hesabı Oluşturuldu: aybuke_admin | Şifre: admin123")
    else:
        print("Veritabanı zaten mevcut ve yönetici hesabı tanımlı.")
    db.close()

if __name__ == "__main__":
    create_initial_admin()