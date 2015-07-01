let assert = (condition, message="Assertion failed") => {
  if (condition)
		return;
	
  throw new Error(message);
}

let benchmark = (fnc) => {
	let t0 = +new Date();
	fnc();
	let t1 = +new Date();
	return t1 - t0;
}

export default {
	assert,
	benchmark
};