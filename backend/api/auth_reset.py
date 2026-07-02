"""
auth_reset.py
GuidaPlate - Forgot password and reset password endpoints
"""
import logging
import os
import secrets
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from sqlalchemy.orm import Session

from backend.auth.security import hash_password
from backend.database.db import PasswordResetToken, User, get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Authentication"])


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


@router.post("/auth/forgot-password")
def forgot_password(body: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()

    if user:
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=1)

        reset_token = PasswordResetToken(
            email=body.email,
            token=token,
            expires_at=expires_at,
        )
        db.add(reset_token)
        db.commit()

        _base = os.getenv("RESET_BASE_URL", "http://localhost:5173")
        reset_url = (
            f"{_base.rstrip('/')}"
            f"/reset-password"
            f"?token={token}"
        )

        message = Mail(
            from_email=os.getenv("SENDGRID_FROM_EMAIL"),
            to_emails=body.email,
            subject="GuidaPlate — Reset your password",
            html_content=f"""
            <div style="font-family: sans-serif;
                        max-width: 480px; margin: 0 auto;">
              <h2 style="color: #0f766e;">GuidaPlate</h2>
              <p>You requested a password reset for your
                 GuidaPlate account.</p>
              <p>Click the button below to reset your
                 password. This link expires in 1 hour.</p>
              <a href="{reset_url}"
                 style="display: inline-block;
                        background: #0f766e;
                        color: white;
                        padding: 12px 24px;
                        border-radius: 6px;
                        text-decoration: none;
                        margin: 16px 0;">
                Reset Password
              </a>
              <p style="color: #666; font-size: 12px;">
                If you didn't request this, ignore this
                email. Your password will not change.
              </p>
              <p style="color: #666; font-size: 12px;">
                ⚕ GuidaPlate — CKD Dietary Guidance Platform
              </p>
            </div>
            """,
        )

        try:
            sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))
            sg.send(message)
        except Exception as e:
            logger.error("SendGrid error: %s", e)
            raise HTTPException(
                status_code=503,
                detail=(
                    "Email service temporarily unavailable. Please try "
                    "again in a few minutes."
                ),
            ) from e

    return {"message": "If this email is registered, a reset link has been sent."}


@router.post("/auth/reset-password")
def reset_password(body: ResetPasswordRequest, db: Session = Depends(get_db)):
    token_record = (
        db.query(PasswordResetToken)
        .filter(PasswordResetToken.token == body.token)
        .first()
    )

    if not token_record:
        raise HTTPException(status_code=400, detail="Invalid or expired reset link")

    if token_record.used:
        raise HTTPException(status_code=400, detail="This reset link has already been used")

    if token_record.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="This reset link has expired")

    user = db.query(User).filter(User.email == token_record.email).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired reset link")

    user.hashed_password = hash_password(body.new_password)
    token_record.used = True
    db.commit()

    return {"message": "Password reset successfully. Please log in."}
