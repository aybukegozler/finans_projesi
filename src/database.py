import os
from pathlib import Path

import bcrypt
from sqlalchemy import Column, Integer, String, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{DATA_DIR / 'quant_app.db'}",
)

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://",
        "postgresql://",
        1,
    )

connect_args = (
    {"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite")
    else {}
)

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="user", nullable=False)


Base.metadata.create_all(bind=engine)


def get_database_backend() -> str:
    """Aktif SQLAlchemy veritabanı backend adını döndürür."""
    return engine.url.get_backend_name()


def check_database_connection() -> bool:
    """Veritabanına basit bir SELECT 1 sorgusu gönderir."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return True

    except Exception as error:
        # Connection string veya parola gibi hassas bilgileri loglamıyoruz.
        print(
            "Database health check failed: "
            f"{type(error).__name__}"
        )

        return False


def get_password_hash(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def create_initial_users() -> int:
    """Environment variable ile tanımlanan kullanıcıları hazırlar."""

    configured_users = [
        (
            os.getenv("ADMIN_USERNAME"),
            os.getenv("ADMIN_PASSWORD"),
            "admin",
        ),
        (
            os.getenv("USER_USERNAME"),
            os.getenv("USER_PASSWORD"),
            "user",
        ),
    ]

    db = SessionLocal()
    prepared_count = 0

    try:
        for username, password, role in configured_users:
            if not username or not password:
                continue

            user = db.query(User).filter(
                User.username == username
            ).first()

            if user is None:
                user = User(
                    username=username,
                    hashed_password=get_password_hash(password),
                    role=role,
                )
                db.add(user)
            else:
                password_matches = bcrypt.checkpw(
                    password.encode("utf-8"),
                    user.hashed_password.encode("utf-8"),
                )

                if not password_matches:
                    user.hashed_password = get_password_hash(password)

                user.role = role

            prepared_count += 1

        db.commit()
        return prepared_count

    finally:
        db.close()


if __name__ == "__main__":
    count = create_initial_users()
    print(f"Hazırlanan environment kullanıcısı: {count}")
