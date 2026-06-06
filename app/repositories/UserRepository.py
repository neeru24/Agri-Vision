from datetime import datetime
from typing import Optional
from models import User, db

class UserRepository:
    @staticmethod
    def get_by_id(user_id: str) -> Optional[User]:
        return db.session.get(User, user_id)

    @staticmethod
    def get_by_email(email: str) -> Optional[User]:
        return User.query.filter_by(email=email).first()

    @staticmethod
    def create(user: User) -> User:
        db.session.add(user)
        db.session.commit()
        return user

    @staticmethod
    def save(user: User) -> User:
        db.session.commit()
        return user

    @staticmethod
    def update_last_login(user: User) -> None:
        user.last_login = datetime.utcnow()
        db.session.commit()

    @staticmethod
    def get_all() -> list[User]:
        return User.query.all()
