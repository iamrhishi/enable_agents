import { useState, useCallback, useMemo } from 'react';

/**
 * useValidation Hook
 *
 * Provides form validation with built-in validators.
 *
 * Usage:
 *   const { values, errors, touched, handleChange, handleBlur, validate, isValid } = useValidation({
 *     email: { value: '', validators: [validators.required(), validators.email()] },
 *     password: { value: '', validators: [validators.required(), validators.minLength(8)] }
 *   });
 *
 *   <input
 *     name="email"
 *     value={values.email}
 *     onChange={handleChange}
 *     onBlur={handleBlur}
 *   />
 *   {touched.email && errors.email && <span>{errors.email}</span>}
 */

// Built-in validators
export const validators = {
  required: (message = 'This field is required') => (value) => {
    if (value === null || value === undefined || value === '' || (Array.isArray(value) && value.length === 0)) {
      return message;
    }
    return null;
  },

  email: (message = 'Please enter a valid email address') => (value) => {
    if (!value) return null;
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(value)) {
      return message;
    }
    return null;
  },

  minLength: (min, message) => (value) => {
    if (!value) return null;
    if (value.length < min) {
      return message || `Must be at least ${min} characters`;
    }
    return null;
  },

  maxLength: (max, message) => (value) => {
    if (!value) return null;
    if (value.length > max) {
      return message || `Must be no more than ${max} characters`;
    }
    return null;
  },

  pattern: (regex, message = 'Invalid format') => (value) => {
    if (!value) return null;
    if (!regex.test(value)) {
      return message;
    }
    return null;
  },

  matches: (fieldName, message) => (value, allValues) => {
    if (!value) return null;
    if (value !== allValues[fieldName]) {
      return message || `Must match ${fieldName}`;
    }
    return null;
  },

  url: (message = 'Please enter a valid URL') => (value) => {
    if (!value) return null;
    try {
      new URL(value);
      return null;
    } catch {
      return message;
    }
  },

  number: (message = 'Must be a number') => (value) => {
    if (!value) return null;
    if (isNaN(Number(value))) {
      return message;
    }
    return null;
  },

  min: (minValue, message) => (value) => {
    if (!value) return null;
    if (Number(value) < minValue) {
      return message || `Must be at least ${minValue}`;
    }
    return null;
  },

  max: (maxValue, message) => (value) => {
    if (!value) return null;
    if (Number(value) > maxValue) {
      return message || `Must be no more than ${maxValue}`;
    }
    return null;
  }
};

function useValidation(schema) {
  // Extract initial values from schema
  const initialValues = useMemo(() => {
    const values = {};
    Object.keys(schema).forEach((key) => {
      values[key] = schema[key].value ?? '';
    });
    return values;
  }, []);

  const [values, setValues] = useState(initialValues);
  const [errors, setErrors] = useState({});
  const [touched, setTouched] = useState({});

  // Validate a single field
  const validateField = useCallback((name, value, allValues) => {
    const fieldSchema = schema[name];
    if (!fieldSchema || !fieldSchema.validators) return null;

    for (const validator of fieldSchema.validators) {
      const error = validator(value, allValues);
      if (error) return error;
    }
    return null;
  }, [schema]);

  // Validate all fields
  const validate = useCallback(() => {
    const newErrors = {};
    let isValid = true;

    Object.keys(schema).forEach((name) => {
      const error = validateField(name, values[name], values);
      if (error) {
        newErrors[name] = error;
        isValid = false;
      }
    });

    setErrors(newErrors);
    // Mark all fields as touched
    const allTouched = {};
    Object.keys(schema).forEach((name) => {
      allTouched[name] = true;
    });
    setTouched(allTouched);

    return isValid;
  }, [schema, values, validateField]);

  // Handle input change
  const handleChange = useCallback((e) => {
    const { name, value, type, checked } = e.target;
    const newValue = type === 'checkbox' ? checked : value;

    setValues((prev) => {
      const newValues = { ...prev, [name]: newValue };

      // Validate on change if field was touched
      if (touched[name]) {
        const error = validateField(name, newValue, newValues);
        setErrors((prevErrors) => ({
          ...prevErrors,
          [name]: error
        }));
      }

      return newValues;
    });
  }, [touched, validateField]);

  // Handle input blur
  const handleBlur = useCallback((e) => {
    const { name, value } = e.target;

    setTouched((prev) => ({ ...prev, [name]: true }));

    const error = validateField(name, value, values);
    setErrors((prev) => ({ ...prev, [name]: error }));
  }, [validateField, values]);

  // Set a specific field value programmatically
  const setValue = useCallback((name, value) => {
    setValues((prev) => ({ ...prev, [name]: value }));
  }, []);

  // Set multiple values
  const setMultipleValues = useCallback((newValues) => {
    setValues((prev) => ({ ...prev, ...newValues }));
  }, []);

  // Reset form to initial values
  const reset = useCallback(() => {
    setValues(initialValues);
    setErrors({});
    setTouched({});
  }, [initialValues]);

  // Check if form is valid (no errors)
  const isValid = useMemo(() => {
    return Object.keys(errors).every((key) => !errors[key]);
  }, [errors]);

  // Check if form is dirty (values changed from initial)
  const isDirty = useMemo(() => {
    return Object.keys(values).some((key) => values[key] !== initialValues[key]);
  }, [values, initialValues]);

  return {
    values,
    errors,
    touched,
    handleChange,
    handleBlur,
    validate,
    setValue,
    setMultipleValues,
    reset,
    isValid,
    isDirty
  };
}

export default useValidation;
