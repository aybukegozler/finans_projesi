import argparse
from getpass import getpass

from src.database import SessionLocal, User, get_password_hash


def create_user(username: str, password: str, role: str) -> None:
    db = SessionLocal()

    try:
        existing_user = db.query(User).filter(
            User.username == username
        ).first()

        if existing_user:
            raise SystemExit(
                f"Kullanıcı zaten mevcut: {username}"
            )

        user = User(
            username=username,
            hashed_password=get_password_hash(password),
            role=role,
        )

        db.add(user)
        db.commit()

        print(f"Kullanıcı oluşturuldu: {username} ({role})")

    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quant Dashboard kullanıcısı oluşturur."
    )

    parser.add_argument(
        "--username",
        required=True,
    )

    parser.add_argument(
        "--role",
        choices=["admin", "user"],
        default="user",
    )

    args = parser.parse_args()

    password = getpass("Şifre: ")
    confirmation = getpass("Şifreyi tekrar yaz: ")

    if password != confirmation:
        raise SystemExit("Şifreler eşleşmiyor.")

    if len(password) < 10:
        raise SystemExit(
            "Şifre en az 10 karakter olmalıdır."
        )

    create_user(
        args.username,
        password,
        args.role,
    )


if __name__ == "__main__":
    main()
