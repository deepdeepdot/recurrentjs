let assert = (condition, message="Assertion failed") => {
  if (condition)
		return;
	
  throw new Error(message);
}

export default {assert};