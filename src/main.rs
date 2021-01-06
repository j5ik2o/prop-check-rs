mod state;

#[derive(Debug)]
struct Person {
  name: &'static str,
  age: i32,
}

fn move_f() {
  println!("----- move -----");
  let a = Person { name: "masuda", age: 50 };
  let a_ptr: *const Person = &a;
  println!("a is {:?}", a);
  println!("a_ptr is {:?}", a_ptr);
  println!("a.name.ptr is {:?}", a.name.as_ptr());
  let b = a;
  let b_ptr: *const Person = &b;
  println!("b is {:?}", b);
  println!("b_ptr is {:?}", b_ptr);
  println!("b.name.ptr is {:?}", b.name.as_ptr());
  // println!("a is {:?}", a)
}

fn borrow_f() {
  println!("----- borrow -----");
  let a = Person { name: "masuda", age: 50 };
  let a_ptr: *const Person = &a;
  println!("a is {:?}", a);
  println!("a_ptr is {:?}", a_ptr);
  println!("a.name.ptr is {:?}", a.name.as_ptr());
  let b = &a;
  let b_ptr: *const Person = b;
  println!("b is {:?}", b);
  println!("b_ptr is {:?}", b_ptr);
  println!("b.name.ptr is {:?}", b.name.as_ptr());
  println!("a is {:?}", a)
}

fn chars() {
  let s = "This is ねこ😸neko 文字列";
  let mut v: Vec<char> = Vec::new();
  for c in s.chars() {
    v.push(c);
  }
  let v = &v[8..15];
  let mut s = String::new();
  for c in v {
    s.push(*c);
  }
  println!("s is {}", s);
}


fn main() {
  chars();
//  move_f();
//  borrow_f();
}