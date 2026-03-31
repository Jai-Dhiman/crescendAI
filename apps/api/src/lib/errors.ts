export class DomainError extends Error {
	constructor(message: string) {
		super(message);
		this.name = this.constructor.name;
	}
}

export class NotFoundError extends DomainError {
	constructor(entity: string, id: string) {
		super(`${entity} not found: ${id}`);
	}
}

export class AuthenticationError extends DomainError {
	constructor(message = "Authentication required") {
		super(message);
	}
}

export class ValidationError extends DomainError {
	constructor(message: string) {
		super(message);
	}
}

export class InferenceError extends DomainError {
	constructor(message: string) {
		super(message);
	}
}
