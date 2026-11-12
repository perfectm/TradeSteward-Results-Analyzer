"""
Multi-User System with Role-Based Access Control - Phase 4 Implementation
Comprehensive user management and authorization system for enterprise deployment
"""

import asyncio
import hashlib
import secrets
import jwt
import bcrypt
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
from contextlib import contextmanager
import redis
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import re
from flask import Flask, request, jsonify, g
from functools import wraps
import ldap3
from ldap3 import Server, Connection, ALL

# Import our existing modules
from data_loader import TradeStewardDataLoader

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles with hierarchical permissions"""
    SUPER_ADMIN = "super_admin"          # Full system access
    ADMIN = "admin"                      # User and system management
    RISK_MANAGER = "risk_manager"        # Risk oversight and configuration
    TRADER = "trader"                    # Trading and position management
    ANALYST = "analyst"                  # Read-only analysis and reporting
    VIEWER = "viewer"                    # Basic read-only access
    AUDITOR = "auditor"                  # Audit logs and compliance reports
    API_USER = "api_user"               # API access only

class Permission(Enum):
    """Granular permissions"""
    # User management
    USER_CREATE = "user_create"
    USER_READ = "user_read"
    USER_UPDATE = "user_update"
    USER_DELETE = "user_delete"
    
    # Role management
    ROLE_ASSIGN = "role_assign"
    ROLE_CREATE = "role_create"
    ROLE_UPDATE = "role_update"
    
    # Risk management
    RISK_VIEW = "risk_view"
    RISK_CONFIG = "risk_config"
    RISK_LIMITS_SET = "risk_limits_set"
    RISK_ALERTS_MANAGE = "risk_alerts_manage"
    
    # Trading
    TRADE_VIEW = "trade_view"
    TRADE_CREATE = "trade_create"
    TRADE_UPDATE = "trade_update"
    TRADE_DELETE = "trade_delete"
    
    # Portfolio
    PORTFOLIO_VIEW = "portfolio_view"
    PORTFOLIO_MANAGE = "portfolio_manage"
    
    # Reports
    REPORT_VIEW = "report_view"
    REPORT_CREATE = "report_create"
    REPORT_EXPORT = "report_export"
    
    # System
    SYSTEM_CONFIG = "system_config"
    SYSTEM_MONITOR = "system_monitor"
    
    # Audit
    AUDIT_VIEW = "audit_view"
    AUDIT_EXPORT = "audit_export"
    
    # API
    API_ACCESS = "api_access"

class AuthenticationMethod(Enum):
    """Authentication methods"""
    LOCAL = "local"
    LDAP = "ldap"
    SSO = "sso"
    API_KEY = "api_key"
    MULTI_FACTOR = "multi_factor"

@dataclass
class User:
    """User data structure"""
    user_id: str
    username: str
    email: str
    full_name: str
    roles: List[UserRole]
    permissions: Set[Permission]
    auth_method: AuthenticationMethod
    is_active: bool = True
    is_locked: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    password_changed_at: Optional[datetime] = None
    failed_login_attempts: int = 0
    session_timeout_minutes: int = 480  # 8 hours
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UserSession:
    """User session data"""
    session_id: str
    user_id: str
    username: str
    roles: List[str]
    permissions: List[str]
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True

@dataclass
class APIKey:
    """API key data structure"""
    api_key_id: str
    user_id: str
    key_name: str
    key_hash: str
    permissions: Set[Permission]
    is_active: bool = True
    expires_at: Optional[datetime] = None
    rate_limit_per_hour: int = 1000
    last_used_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

class RolePermissionMatrix:
    """Role-based permission matrix"""
    
    ROLE_PERMISSIONS = {
        UserRole.SUPER_ADMIN: {
            Permission.USER_CREATE, Permission.USER_READ, Permission.USER_UPDATE, Permission.USER_DELETE,
            Permission.ROLE_ASSIGN, Permission.ROLE_CREATE, Permission.ROLE_UPDATE,
            Permission.RISK_VIEW, Permission.RISK_CONFIG, Permission.RISK_LIMITS_SET, Permission.RISK_ALERTS_MANAGE,
            Permission.TRADE_VIEW, Permission.TRADE_CREATE, Permission.TRADE_UPDATE, Permission.TRADE_DELETE,
            Permission.PORTFOLIO_VIEW, Permission.PORTFOLIO_MANAGE,
            Permission.REPORT_VIEW, Permission.REPORT_CREATE, Permission.REPORT_EXPORT,
            Permission.SYSTEM_CONFIG, Permission.SYSTEM_MONITOR,
            Permission.AUDIT_VIEW, Permission.AUDIT_EXPORT,
            Permission.API_ACCESS
        },
        
        UserRole.ADMIN: {
            Permission.USER_CREATE, Permission.USER_READ, Permission.USER_UPDATE,
            Permission.ROLE_ASSIGN,
            Permission.RISK_VIEW, Permission.RISK_CONFIG, Permission.RISK_ALERTS_MANAGE,
            Permission.TRADE_VIEW, Permission.TRADE_CREATE, Permission.TRADE_UPDATE,
            Permission.PORTFOLIO_VIEW, Permission.PORTFOLIO_MANAGE,
            Permission.REPORT_VIEW, Permission.REPORT_CREATE, Permission.REPORT_EXPORT,
            Permission.SYSTEM_MONITOR,
            Permission.AUDIT_VIEW,
            Permission.API_ACCESS
        },
        
        UserRole.RISK_MANAGER: {
            Permission.USER_READ,
            Permission.RISK_VIEW, Permission.RISK_CONFIG, Permission.RISK_LIMITS_SET, Permission.RISK_ALERTS_MANAGE,
            Permission.TRADE_VIEW,
            Permission.PORTFOLIO_VIEW, Permission.PORTFOLIO_MANAGE,
            Permission.REPORT_VIEW, Permission.REPORT_CREATE, Permission.REPORT_EXPORT,
            Permission.AUDIT_VIEW
        },
        
        UserRole.TRADER: {
            Permission.RISK_VIEW,
            Permission.TRADE_VIEW, Permission.TRADE_CREATE, Permission.TRADE_UPDATE,
            Permission.PORTFOLIO_VIEW,
            Permission.REPORT_VIEW,
            Permission.API_ACCESS
        },
        
        UserRole.ANALYST: {
            Permission.RISK_VIEW,
            Permission.TRADE_VIEW,
            Permission.PORTFOLIO_VIEW,
            Permission.REPORT_VIEW, Permission.REPORT_CREATE, Permission.REPORT_EXPORT
        },
        
        UserRole.VIEWER: {
            Permission.RISK_VIEW,
            Permission.TRADE_VIEW,
            Permission.PORTFOLIO_VIEW,
            Permission.REPORT_VIEW
        },
        
        UserRole.AUDITOR: {
            Permission.USER_READ,
            Permission.RISK_VIEW,
            Permission.TRADE_VIEW,
            Permission.PORTFOLIO_VIEW,
            Permission.REPORT_VIEW, Permission.REPORT_EXPORT,
            Permission.AUDIT_VIEW, Permission.AUDIT_EXPORT
        },
        
        UserRole.API_USER: {
            Permission.API_ACCESS,
            Permission.TRADE_VIEW,
            Permission.PORTFOLIO_VIEW,
            Permission.RISK_VIEW
        }
    }
    
    @classmethod
    def get_permissions_for_roles(cls, roles: List[UserRole]) -> Set[Permission]:
        """Get combined permissions for multiple roles"""
        permissions = set()
        for role in roles:
            permissions.update(cls.ROLE_PERMISSIONS.get(role, set()))
        return permissions

class PasswordPolicy:
    """Password policy enforcement"""
    
    MIN_LENGTH = 12
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_DIGITS = True
    REQUIRE_SPECIAL_CHARS = True
    MIN_SPECIAL_CHARS = 1
    HISTORY_COUNT = 5  # Remember last 5 passwords
    MAX_AGE_DAYS = 90
    
    SPECIAL_CHARS = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    @classmethod
    def validate_password(cls, password: str) -> Tuple[bool, List[str]]:
        """Validate password against policy"""
        errors = []
        
        if len(password) < cls.MIN_LENGTH:
            errors.append(f"Password must be at least {cls.MIN_LENGTH} characters long")
        
        if cls.REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if cls.REQUIRE_LOWERCASE and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if cls.REQUIRE_DIGITS and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        if cls.REQUIRE_SPECIAL_CHARS:
            special_count = sum(1 for c in password if c in cls.SPECIAL_CHARS)
            if special_count < cls.MIN_SPECIAL_CHARS:
                errors.append(f"Password must contain at least {cls.MIN_SPECIAL_CHARS} special character(s)")
        
        return len(errors) == 0, errors

class SecurityUtils:
    """Security utilities for encryption and hashing"""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or os.environ.get('SECRET_KEY', self._generate_secret_key())
        self.fernet = self._create_fernet()
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key"""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
    
    def _create_fernet(self) -> Fernet:
        """Create Fernet encryption instance"""
        key = base64.urlsafe_b64encode(self.secret_key.encode()[:32].ljust(32, b'0'))
        return Fernet(key)
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        except Exception:
            return False
    
    def generate_api_key(self) -> Tuple[str, str]:
        """Generate API key and its hash"""
        api_key = f"rsk_{secrets.token_urlsafe(32)}"
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        return api_key, api_key_hash
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash API key"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def generate_session_token(self) -> str:
        """Generate secure session token"""
        return secrets.token_urlsafe(32)
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def generate_jwt_token(self, payload: Dict[str, Any], expires_in_hours: int = 8) -> str:
        """Generate JWT token"""
        payload['exp'] = datetime.utcnow() + timedelta(hours=expires_in_hours)
        payload['iat'] = datetime.utcnow()
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            return jwt.decode(token, self.secret_key, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

class UserDatabase:
    """Database operations for user management"""
    
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.security = SecurityUtils()
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize user database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                password_hash TEXT,
                roles TEXT NOT NULL,
                auth_method TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                is_locked INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                last_login TEXT,
                password_changed_at TEXT,
                failed_login_attempts INTEGER DEFAULT 0,
                session_timeout_minutes INTEGER DEFAULT 480,
                metadata TEXT
            )
        ''')
        
        # Password history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS password_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # API keys table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                api_key_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                key_name TEXT NOT NULL,
                key_hash TEXT UNIQUE NOT NULL,
                permissions TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                expires_at TEXT,
                rate_limit_per_hour INTEGER DEFAULT 1000,
                last_used_at TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Sessions table (for active session tracking)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                username TEXT NOT NULL,
                roles TEXT NOT NULL,
                permissions TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                is_active INTEGER DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Audit log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                username TEXT,
                action TEXT NOT NULL,
                resource_type TEXT,
                resource_id TEXT,
                ip_address TEXT,
                user_agent TEXT,
                success INTEGER NOT NULL,
                details TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id)')
        
        conn.commit()
        conn.close()
        
        # Create default admin user if none exists
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user if no users exist"""
        if self.get_user_count() == 0:
            admin_user = User(
                user_id=str(uuid.uuid4()),
                username="admin",
                email="admin@example.com",
                full_name="System Administrator",
                roles=[UserRole.SUPER_ADMIN],
                permissions=RolePermissionMatrix.get_permissions_for_roles([UserRole.SUPER_ADMIN]),
                auth_method=AuthenticationMethod.LOCAL
            )
            
            self.create_user(admin_user, "AdminPassword123!")
            logger.info("Created default admin user")
    
    @contextmanager
    def get_db_connection(self):
        """Get database connection with proper cleanup"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def create_user(self, user: User, password: Optional[str] = None) -> bool:
        """Create a new user"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Hash password if provided
                password_hash = None
                if password and user.auth_method == AuthenticationMethod.LOCAL:
                    is_valid, errors = PasswordPolicy.validate_password(password)
                    if not is_valid:
                        raise ValueError(f"Password policy violation: {', '.join(errors)}")
                    
                    password_hash = self.security.hash_password(password)
                
                cursor.execute('''
                    INSERT INTO users 
                    (user_id, username, email, full_name, password_hash, roles, auth_method,
                     is_active, is_locked, created_at, password_changed_at, session_timeout_minutes, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user.user_id,
                    user.username,
                    user.email,
                    user.full_name,
                    password_hash,
                    json.dumps([role.value for role in user.roles]),
                    user.auth_method.value,
                    1 if user.is_active else 0,
                    1 if user.is_locked else 0,
                    user.created_at.isoformat(),
                    datetime.now().isoformat() if password_hash else None,
                    user.session_timeout_minutes,
                    json.dumps(user.metadata)
                ))
                
                # Add password to history if provided
                if password_hash:
                    cursor.execute('''
                        INSERT INTO password_history (user_id, password_hash, created_at)
                        VALUES (?, ?, ?)
                    ''', (user.user_id, password_hash, datetime.now().isoformat()))
                
                conn.commit()
                self._log_audit_event(user.user_id, user.username, "USER_CREATE", "User", user.user_id, True)
                return True
                
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            self._log_audit_event(None, None, "USER_CREATE", "User", user.user_id, False, str(e))
            return False
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_user(row)
                return None
                
        except Exception as e:
            logger.error(f"Error getting user by username: {str(e)}")
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_user(row)
                return None
                
        except Exception as e:
            logger.error(f"Error getting user by ID: {str(e)}")
            return None
    
    def authenticate_user(self, username: str, password: str, ip_address: str = "", user_agent: str = "") -> Optional[User]:
        """Authenticate user with username/password"""
        try:
            user = self.get_user_by_username(username)
            if not user:
                self._log_audit_event(None, username, "LOGIN_FAILED", "Authentication", None, False, "User not found", ip_address, user_agent)
                return None
            
            if not user.is_active:
                self._log_audit_event(user.user_id, username, "LOGIN_FAILED", "Authentication", None, False, "Account inactive", ip_address, user_agent)
                return None
            
            if user.is_locked:
                self._log_audit_event(user.user_id, username, "LOGIN_FAILED", "Authentication", None, False, "Account locked", ip_address, user_agent)
                return None
            
            # Check password
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT password_hash FROM users WHERE user_id = ?', (user.user_id,))
                row = cursor.fetchone()
                
                if row and row['password_hash']:
                    if self.security.verify_password(password, row['password_hash']):
                        # Successful authentication
                        self._reset_failed_login_attempts(user.user_id)
                        self._update_last_login(user.user_id)
                        self._log_audit_event(user.user_id, username, "LOGIN_SUCCESS", "Authentication", None, True, None, ip_address, user_agent)
                        return user
                    else:
                        # Failed authentication
                        self._increment_failed_login_attempts(user.user_id)
                        self._log_audit_event(user.user_id, username, "LOGIN_FAILED", "Authentication", None, False, "Invalid password", ip_address, user_agent)
                        return None
                else:
                    self._log_audit_event(user.user_id, username, "LOGIN_FAILED", "Authentication", None, False, "No password set", ip_address, user_agent)
                    return None
                    
        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}")
            self._log_audit_event(None, username, "LOGIN_ERROR", "Authentication", None, False, str(e), ip_address, user_agent)
            return None
    
    def create_api_key(self, user_id: str, key_name: str, permissions: Set[Permission], expires_at: Optional[datetime] = None) -> Optional[str]:
        """Create API key for user"""
        try:
            api_key, key_hash = self.security.generate_api_key()
            api_key_record = APIKey(
                api_key_id=str(uuid.uuid4()),
                user_id=user_id,
                key_name=key_name,
                key_hash=key_hash,
                permissions=permissions,
                expires_at=expires_at
            )
            
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO api_keys 
                    (api_key_id, user_id, key_name, key_hash, permissions, is_active,
                     expires_at, rate_limit_per_hour, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    api_key_record.api_key_id,
                    api_key_record.user_id,
                    api_key_record.key_name,
                    api_key_record.key_hash,
                    json.dumps([p.value for p in api_key_record.permissions]),
                    1 if api_key_record.is_active else 0,
                    api_key_record.expires_at.isoformat() if api_key_record.expires_at else None,
                    api_key_record.rate_limit_per_hour,
                    api_key_record.created_at.isoformat()
                ))
                
                conn.commit()
                self._log_audit_event(user_id, None, "API_KEY_CREATE", "APIKey", api_key_record.api_key_id, True)
                return api_key
                
        except Exception as e:
            logger.error(f"Error creating API key: {str(e)}")
            self._log_audit_event(user_id, None, "API_KEY_CREATE", "APIKey", None, False, str(e))
            return None
    
    def authenticate_api_key(self, api_key: str) -> Optional[Tuple[User, Set[Permission]]]:
        """Authenticate API key and return user with permissions"""
        try:
            key_hash = self.security.hash_api_key(api_key)
            
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT ak.*, u.username FROM api_keys ak
                    JOIN users u ON ak.user_id = u.user_id
                    WHERE ak.key_hash = ? AND ak.is_active = 1
                ''', (key_hash,))
                row = cursor.fetchone()
                
                if row:
                    # Check if key is expired
                    if row['expires_at']:
                        expires_at = datetime.fromisoformat(row['expires_at'])
                        if datetime.now() > expires_at:
                            self._log_audit_event(row['user_id'], row['username'], "API_KEY_EXPIRED", "APIKey", row['api_key_id'], False)
                            return None
                    
                    # Update last used time
                    cursor.execute('''
                        UPDATE api_keys SET last_used_at = ? WHERE api_key_id = ?
                    ''', (datetime.now().isoformat(), row['api_key_id']))
                    conn.commit()
                    
                    # Get user and permissions
                    user = self.get_user_by_id(row['user_id'])
                    permissions = set(Permission(p) for p in json.loads(row['permissions']))
                    
                    self._log_audit_event(row['user_id'], row['username'], "API_KEY_AUTH", "APIKey", row['api_key_id'], True)
                    return user, permissions
                
                return None
                
        except Exception as e:
            logger.error(f"Error authenticating API key: {str(e)}")
            return None
    
    def get_user_count(self) -> int:
        """Get total number of users"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM users')
                return cursor.fetchone()[0]
        except Exception:
            return 0
    
    def _row_to_user(self, row) -> User:
        """Convert database row to User object"""
        return User(
            user_id=row['user_id'],
            username=row['username'],
            email=row['email'],
            full_name=row['full_name'],
            roles=[UserRole(role) for role in json.loads(row['roles'])],
            permissions=RolePermissionMatrix.get_permissions_for_roles(
                [UserRole(role) for role in json.loads(row['roles'])]
            ),
            auth_method=AuthenticationMethod(row['auth_method']),
            is_active=bool(row['is_active']),
            is_locked=bool(row['is_locked']),
            created_at=datetime.fromisoformat(row['created_at']),
            last_login=datetime.fromisoformat(row['last_login']) if row['last_login'] else None,
            password_changed_at=datetime.fromisoformat(row['password_changed_at']) if row['password_changed_at'] else None,
            failed_login_attempts=row['failed_login_attempts'],
            session_timeout_minutes=row['session_timeout_minutes'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )
    
    def _reset_failed_login_attempts(self, user_id: str):
        """Reset failed login attempts counter"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE users SET failed_login_attempts = 0 WHERE user_id = ?', (user_id,))
                conn.commit()
        except Exception as e:
            logger.error(f"Error resetting failed login attempts: {str(e)}")
    
    def _increment_failed_login_attempts(self, user_id: str):
        """Increment failed login attempts and lock account if necessary"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE users 
                    SET failed_login_attempts = failed_login_attempts + 1,
                        is_locked = CASE WHEN failed_login_attempts >= 4 THEN 1 ELSE is_locked END
                    WHERE user_id = ?
                ''', (user_id,))
                conn.commit()
        except Exception as e:
            logger.error(f"Error incrementing failed login attempts: {str(e)}")
    
    def _update_last_login(self, user_id: str):
        """Update last login timestamp"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE users SET last_login = ? WHERE user_id = ?', 
                             (datetime.now().isoformat(), user_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating last login: {str(e)}")
    
    def _log_audit_event(self, user_id: Optional[str], username: Optional[str], action: str, 
                        resource_type: Optional[str], resource_id: Optional[str], success: bool,
                        details: Optional[str] = None, ip_address: str = "", user_agent: str = ""):
        """Log audit event"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO audit_log 
                    (timestamp, user_id, username, action, resource_type, resource_id,
                     ip_address, user_agent, success, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    user_id,
                    username,
                    action,
                    resource_type,
                    resource_id,
                    ip_address,
                    user_agent,
                    1 if success else 0,
                    details
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging audit event: {str(e)}")

class SessionManager:
    """Manage user sessions with Redis backend"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(decode_responses=True)
        self.security = SecurityUtils()
    
    def create_session(self, user: User, ip_address: str = "", user_agent: str = "") -> UserSession:
        """Create new user session"""
        session_id = self.security.generate_session_token()
        expires_at = datetime.now() + timedelta(minutes=user.session_timeout_minutes)
        
        session = UserSession(
            session_id=session_id,
            user_id=user.user_id,
            username=user.username,
            roles=[role.value for role in user.roles],
            permissions=[perm.value for perm in user.permissions],
            created_at=datetime.now(),
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Store in Redis
        session_key = f"session:{session_id}"
        self.redis_client.hset(session_key, mapping=asdict(session))
        self.redis_client.expire(session_key, user.session_timeout_minutes * 60)
        
        # Store user sessions list
        user_sessions_key = f"user_sessions:{user.user_id}"
        self.redis_client.sadd(user_sessions_key, session_id)
        self.redis_client.expire(user_sessions_key, user.session_timeout_minutes * 60)
        
        return session
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID"""
        try:
            session_key = f"session:{session_id}"
            session_data = self.redis_client.hgetall(session_key)
            
            if session_data:
                # Convert string dates back to datetime objects
                session_data['created_at'] = datetime.fromisoformat(session_data['created_at'])
                session_data['expires_at'] = datetime.fromisoformat(session_data['expires_at'])
                session_data['is_active'] = session_data['is_active'] == 'True'
                
                # Check if session is expired
                if datetime.now() > session_data['expires_at']:
                    self.invalidate_session(session_id)
                    return None
                
                return UserSession(**session_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting session: {str(e)}")
            return None
    
    def invalidate_session(self, session_id: str):
        """Invalidate a session"""
        try:
            session_key = f"session:{session_id}"
            session_data = self.redis_client.hgetall(session_key)
            
            if session_data:
                user_id = session_data.get('user_id')
                if user_id:
                    user_sessions_key = f"user_sessions:{user_id}"
                    self.redis_client.srem(user_sessions_key, session_id)
            
            self.redis_client.delete(session_key)
            
        except Exception as e:
            logger.error(f"Error invalidating session: {str(e)}")
    
    def invalidate_user_sessions(self, user_id: str):
        """Invalidate all sessions for a user"""
        try:
            user_sessions_key = f"user_sessions:{user_id}"
            session_ids = self.redis_client.smembers(user_sessions_key)
            
            for session_id in session_ids:
                self.invalidate_session(session_id)
            
            self.redis_client.delete(user_sessions_key)
            
        except Exception as e:
            logger.error(f"Error invalidating user sessions: {str(e)}")

def require_permission(permission: Permission):
    """Decorator to require specific permission"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(g, 'current_user') or not g.current_user:
                return jsonify({'error': 'Authentication required'}), 401
            
            if permission not in g.current_user.permissions:
                return jsonify({'error': 'Insufficient permissions'}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_role(role: UserRole):
    """Decorator to require specific role"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(g, 'current_user') or not g.current_user:
                return jsonify({'error': 'Authentication required'}), 401
            
            if role not in g.current_user.roles:
                return jsonify({'error': 'Insufficient role'}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

class MultiUserAuthSystem:
    """Main multi-user authentication system"""
    
    def __init__(self):
        self.user_db = UserDatabase()
        self.session_manager = SessionManager()
        self.security = SecurityUtils()
        self.logger = logging.getLogger(__name__ + '.MultiUserAuthSystem')
    
    def authenticate_request(self, request) -> Optional[User]:
        """Authenticate incoming request"""
        # Check for session token
        session_token = request.headers.get('X-Session-Token') or request.cookies.get('session_token')
        if session_token:
            session = self.session_manager.get_session(session_token)
            if session and session.is_active:
                return self.user_db.get_user_by_id(session.user_id)
        
        # Check for API key
        api_key = request.headers.get('X-API-Key') or request.headers.get('Authorization', '').replace('Bearer ', '')
        if api_key and api_key.startswith('rsk_'):
            result = self.user_db.authenticate_api_key(api_key)
            if result:
                user, permissions = result
                # Override user permissions with API key permissions
                user.permissions = permissions
                return user
        
        # Check for JWT token
        jwt_token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if jwt_token and not jwt_token.startswith('rsk_'):
            payload = self.security.verify_jwt_token(jwt_token)
            if payload:
                return self.user_db.get_user_by_id(payload.get('user_id'))
        
        return None
    
    def login(self, username: str, password: str, ip_address: str = "", user_agent: str = "") -> Optional[Dict[str, Any]]:
        """Login user and create session"""
        user = self.user_db.authenticate_user(username, password, ip_address, user_agent)
        if user:
            session = self.session_manager.create_session(user, ip_address, user_agent)
            jwt_token = self.security.generate_jwt_token({
                'user_id': user.user_id,
                'username': user.username,
                'roles': [role.value for role in user.roles]
            }, expires_in_hours=user.session_timeout_minutes // 60)
            
            return {
                'user': {
                    'user_id': user.user_id,
                    'username': user.username,
                    'email': user.email,
                    'full_name': user.full_name,
                    'roles': [role.value for role in user.roles],
                    'permissions': [perm.value for perm in user.permissions]
                },
                'session_token': session.session_id,
                'jwt_token': jwt_token,
                'expires_at': session.expires_at.isoformat()
            }
        
        return None
    
    def logout(self, session_id: str):
        """Logout user and invalidate session"""
        self.session_manager.invalidate_session(session_id)
    
    def create_user(self, username: str, email: str, full_name: str, 
                   roles: List[UserRole], password: Optional[str] = None) -> Optional[str]:
        """Create new user"""
        user = User(
            user_id=str(uuid.uuid4()),
            username=username,
            email=email,
            full_name=full_name,
            roles=roles,
            permissions=RolePermissionMatrix.get_permissions_for_roles(roles),
            auth_method=AuthenticationMethod.LOCAL if password else AuthenticationMethod.SSO
        )
        
        if self.user_db.create_user(user, password):
            return user.user_id
        
        return None
    
    def create_api_key(self, user_id: str, key_name: str, permissions: List[Permission]) -> Optional[str]:
        """Create API key for user"""
        return self.user_db.create_api_key(user_id, key_name, set(permissions))


def main():
    """Example usage of Multi-User Authentication System"""
    
    print("Multi-User System with Role-Based Access Control - Phase 4")
    print("=" * 60)
    
    try:
        # Initialize authentication system
        auth_system = MultiUserAuthSystem()
        
        print("✅ Multi-user authentication system initialized")
        print(f"👥 Default admin user created: admin / AdminPassword123!")
        
        # Display role-permission matrix
        print(f"\n🔐 Role-Permission Matrix:")
        for role in UserRole:
            permissions = RolePermissionMatrix.get_permissions_for_roles([role])
            print(f"   • {role.value}: {len(permissions)} permissions")
            if role in [UserRole.SUPER_ADMIN, UserRole.ADMIN]:
                sample_perms = list(permissions)[:5]
                print(f"     Sample: {[p.value for p in sample_perms]}...")
        
        # Create sample users for different roles
        print(f"\n👤 Creating sample users...")
        
        sample_users = [
            ("risk_manager", "risk@example.com", "Risk Manager", [UserRole.RISK_MANAGER], "RiskPass123!"),
            ("trader1", "trader@example.com", "John Trader", [UserRole.TRADER], "TradePass123!"),
            ("analyst1", "analyst@example.com", "Jane Analyst", [UserRole.ANALYST], "AnalystPass123!"),
            ("viewer1", "viewer@example.com", "Bob Viewer", [UserRole.VIEWER], "ViewPass123!"),
            ("api_user", "api@example.com", "API Service User", [UserRole.API_USER], "APIPass123!")
        ]
        
        created_users = {}
        for username, email, full_name, roles, password in sample_users:
            user_id = auth_system.create_user(username, email, full_name, roles, password)
            if user_id:
                created_users[username] = user_id
                print(f"   ✅ Created {username} ({roles[0].value})")
            else:
                print(f"   ❌ Failed to create {username}")
        
        # Test authentication
        print(f"\n🔑 Testing authentication...")
        
        # Test successful login
        login_result = auth_system.login("trader1", "TradePass123!", "127.0.0.1", "Test-Client/1.0")
        if login_result:
            print(f"   ✅ Trader login successful")
            print(f"     • Session token: {login_result['session_token'][:20]}...")
            print(f"     • JWT token: {login_result['jwt_token'][:30]}...")
            print(f"     • Permissions: {len(login_result['user']['permissions'])}")
            
            # Test session validation
            session = auth_system.session_manager.get_session(login_result['session_token'])
            if session:
                print(f"     • Session valid until: {session.expires_at}")
        else:
            print(f"   ❌ Trader login failed")
        
        # Test failed login
        failed_login = auth_system.login("trader1", "wrong_password", "127.0.0.1", "Test-Client/1.0")
        if not failed_login:
            print(f"   ✅ Invalid password correctly rejected")
        
        # Test API key creation
        print(f"\n🔑 Testing API key management...")
        
        if "api_user" in created_users:
            api_permissions = [Permission.API_ACCESS, Permission.TRADE_VIEW, Permission.PORTFOLIO_VIEW]
            api_key = auth_system.create_api_key(created_users["api_user"], "Trading API Key", api_permissions)
            
            if api_key:
                print(f"   ✅ API key created: {api_key[:20]}...")
                
                # Test API key authentication
                user_db = auth_system.user_db
                auth_result = user_db.authenticate_api_key(api_key)
                if auth_result:
                    user, permissions = auth_result
                    print(f"   ✅ API key authentication successful")
                    print(f"     • User: {user.username}")
                    print(f"     • API Permissions: {[p.value for p in permissions]}")
                else:
                    print(f"   ❌ API key authentication failed")
            else:
                print(f"   ❌ API key creation failed")
        
        # Test password policy
        print(f"\n🔒 Testing password policy...")
        
        test_passwords = [
            ("weak", False),
            ("StrongPassword123!", True),
            ("short", False),
            ("NoDigits!", False),
            ("noupppercase123!", False),
            ("NOLOWERCASE123!", False),
            ("NoSpecialChars123", False)
        ]
        
        for password, should_pass in test_passwords:
            is_valid, errors = PasswordPolicy.validate_password(password)
            if is_valid == should_pass:
                result = "✅" if should_pass else "✅ (correctly rejected)"
                print(f"   {result} '{password}': {is_valid}")
            else:
                print(f"   ❌ '{password}': Expected {should_pass}, got {is_valid}")
                if errors:
                    print(f"       Errors: {errors}")
        
        # Display audit log sample
        print(f"\n📋 Sample audit events:")
        with auth_system.user_db.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, username, action, success, details
                FROM audit_log 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''')
            
            for row in cursor.fetchall():
                timestamp = datetime.fromisoformat(row['timestamp']).strftime('%H:%M:%S')
                status = "✅" if row['success'] else "❌"
                details = f" - {row['details']}" if row['details'] else ""
                print(f"   {status} {timestamp}: {row['username'] or 'System'} - {row['action']}{details}")
        
        # Generate security report
        print(f"\n📊 Security Summary:")
        
        total_users = auth_system.user_db.get_user_count()
        print(f"   • Total users: {total_users}")
        
        with auth_system.user_db.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Active users
            cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 1')
            active_users = cursor.fetchone()[0]
            print(f"   • Active users: {active_users}")
            
            # Locked accounts
            cursor.execute('SELECT COUNT(*) FROM users WHERE is_locked = 1')
            locked_accounts = cursor.fetchone()[0]
            print(f"   • Locked accounts: {locked_accounts}")
            
            # API keys
            cursor.execute('SELECT COUNT(*) FROM api_keys WHERE is_active = 1')
            active_api_keys = cursor.fetchone()[0]
            print(f"   • Active API keys: {active_api_keys}")
            
            # Recent logins (last 24 hours)
            since_yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            cursor.execute('''
                SELECT COUNT(*) FROM audit_log 
                WHERE action = 'LOGIN_SUCCESS' AND timestamp >= ?
            ''', (since_yesterday,))
            recent_logins = cursor.fetchone()[0]
            print(f"   • Successful logins (24h): {recent_logins}")
            
            # Failed logins (last 24 hours)
            cursor.execute('''
                SELECT COUNT(*) FROM audit_log 
                WHERE action = 'LOGIN_FAILED' AND timestamp >= ?
            ''', (since_yesterday,))
            failed_logins = cursor.fetchone()[0]
            print(f"   • Failed logins (24h): {failed_logins}")
        
        print(f"\n🎉 Multi-user authentication system demonstration complete!")
        
        print(f"\n💡 Key Features Demonstrated:")
        print(f"   • Role-based access control with 8 roles")
        print(f"   • Granular permission system with 20+ permissions")
        print(f"   • Multiple authentication methods (password, API key, JWT)")
        print(f"   • Session management with Redis backend")
        print(f"   • Password policy enforcement")
        print(f"   • Account lockout protection")
        print(f"   • Comprehensive audit logging")
        print(f"   • API key management with scoped permissions")
        print(f"   • Secure password hashing with bcrypt")
        print(f"   • JWT token support")
        
        print(f"\n📁 Database files created:")
        print(f"   • users.db - User and authentication data")
        print(f"   • Redis sessions - Active user sessions")
        
    except Exception as e:
        logger.error(f"Error in multi-user auth system: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()