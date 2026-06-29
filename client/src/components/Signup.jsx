import React, { useState } from 'react';

const initialFormData = {
  username: '',
  college: '',
  year: '',
  email: '',
  password: '',
  confirmPassword: '',
};

const Signup = () => {
  const [formData, setFormData] = useState(initialFormData);
  const [errors, setErrors] = useState({});

  const handleChange = (field) => (e) => {
    const { value } = e.target;
    setFormData((prev) => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors((prev) => ({ ...prev, [field]: '' }));
    }
  };

  const validateUsername = (username) => {
    if (!username.trim()) return 'Username is required';
    if (username.trim().length < 3) return 'Username must be at least 3 characters';
    return '';
  };

  const handleUsernameChange = (e) => {
    const { value } = e.target;
    setFormData((prev) => ({ ...prev, username: value }));
    const error = validateUsername(value);
    setErrors((prev) => ({ ...prev, username: error }));
  };

  const validateCollege = (college) => {
    if (!college.trim()) return 'College name is required';
    return '';
  };

  const handleCollegeChange = (e) => {
    const { value } = e.target;
    setFormData((prev) => ({ ...prev, college: value }));
    const error = validateCollege(value);
    setErrors((prev) => ({ ...prev, college: error }));
  };

  const validateYear = (year) => {
    if (!year) return 'Year is required';
    if (isNaN(year) || year < 1 || year > 6) return 'Please enter a valid year (1-6)';
    return '';
  };

  const handleYearChange = (e) => {
    const { value } = e.target;
    setFormData((prev) => ({ ...prev, year: value }));
    const error = validateYear(value);
    setErrors((prev) => ({ ...prev, year: error }));
  };

  const validateEmail = (email) => {
    if (!email.trim()) return 'Email is required';
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) return 'Please enter a valid email address';
    return '';
  };

  const handleEmailChange = (e) => {
    const { value } = e.target;
    setFormData((prev) => ({ ...prev, email: value }));
    const error = validateEmail(value);
    setErrors((prev) => ({ ...prev, email: error }));
  };

  const validatePassword = (password) => {
    if (!password) return 'Password is required';
    if (password.length < 8) return 'Password must be at least 8 characters';
    if (!/[A-Z]/.test(password)) return 'Password must contain an uppercase letter';
    if (!/[a-z]/.test(password)) return 'Password must contain a lowercase letter';
    if (!/\d/.test(password)) return 'Password must contain a number';
    return ;
  };

  const handlePasswordChange = (e) => {
    const { value } = e.target;
    setFormData((prev) => ({ ...prev, password: value }));
    const error = validatePassword(value);
    setErrors((prev) => ({ ...prev, password: error }));
    if (formData.confirmPassword && value !== formData.confirmPassword) {
      setErrors((prev) => ({ ...prev, confirmPassword: 'Passwords do not match' }));
    } else if (formData.confirmPassword) {
      setErrors((prev) => ({ ...prev, confirmPassword: '' }));
    }
  };

  const validateConfirmPassword = (confirmPassword) => {
    if (!confirmPassword) return 'Please confirm your password';
    if (confirmPassword !== formData.password) return 'Passwords do not match';
    return '';
  };

  const handleConfirmPasswordChange = (e) => {
    const { value } = e.target;
    setFormData((prev) => ({ ...prev, confirmPassword: value }));
    const error = validateConfirmPassword(value);
    setErrors((prev) => ({ ...prev, confirmPassword: error }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const newErrors = {
      username: validateUsername(formData.username),
      college: validateCollege(formData.college),
      year: validateYear(formData.year),
      email: validateEmail(formData.email),
      password: validatePassword(formData.password),
      confirmPassword: validateConfirmPassword(formData.confirmPassword),
    };
    setErrors(newErrors);
    const hasErrors = Object.values(newErrors).some((error) => error);
    if (!hasErrors) {
      console.log('Form submitted:', formData);
    }
  };

  return (
    <div className="signup-container">
      <h1>Create Account</h1>
      <p>Join us to get started</p>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="username">Username</label>
          <input
            type="text"
            id="username"
            value={formData.username}
            onChange={handleUsernameChange}
            placeholder="Enter your username"
          />
          {errors.username && <span className="error">{errors.username}</span>}
        </div>

        <div className="form-group">
          <label htmlFor="college">College</label>
          <input
            type="text"
            id="college"
            value={formData.college}
            onChange={handleCollegeChange}
            placeholder="Enter your college name"
          />
          {errors.college && <span className="error">{errors.college}</span>}
        </div>

        <div className="form-group">
          <label htmlFor="year">Year</label>
          <input
            type="number"
            id="year"
            value={formData.year}
            onChange={handleYearChange}
            placeholder="Enter your year (1-6)"
            min="1"
            max="6"
          />
          {errors.year && <span className="error">{errors.year}</span>}
        </div>

        <div className="form-group">
          <label htmlFor="email">Email</label>
          <input
            type="email"
            id="email"
            value={formData.email}
            onChange={handleEmailChange}
            placeholder="Enter your email"
          />
          {errors.email && <span className="error">{errors.email}</span>}
        </div>

        <div className="form-group">
          <label htmlFor="password">Password</label>
          <input
            type="password"
            id="password"
            value={formData.password}
            onChange={handlePasswordChange}
            placeholder="Create a password"
          />
          {errors.password && <span className="error">{errors.password}</span>}
        </div>

        <div className="form-group">
          <label htmlFor="confirmPassword">Confirm Password</label>
          <input
            type="password"
            id="confirmPassword"
            value={formData.confirmPassword}
            onChange={handleConfirmPasswordChange}
            placeholder="Confirm your password"
          />
          {errors.confirmPassword && <span className="error">{errors.confirmPassword}</span>}
        </div>

        <button type="submit" className="btn-submit">
          Create Account
        </button>
      </form>
    </div>
  );
};

export default Signup;
