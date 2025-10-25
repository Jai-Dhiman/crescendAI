pub mod authz;
pub mod jwt;
pub mod middleware;

pub use authz::{
    can_access_content, is_teacher_of_student, require_admin, require_project_access,
    require_role, require_teacher, require_teacher_student_relationship,
};
pub use jwt::{decode_jwt, JwtClaims};
pub use middleware::auth_required;
