from models import User, db

class UserRepository:
    @staticmethod
    def get_by_id(user_id: str) -> User:
        return User.query.get(user_id)

    @staticmethod
    def get_by_email(email: str) -> User:
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
        import datetime
        user.last_login = datetime.datetime.utcnow()
        db.session.commit()

    @staticmethod
    def get_all() -> list[User]:
        return User.query.all()
