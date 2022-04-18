#lang racket

(define (file-lines->list path)
  (call-with-input-file path
    (λ (file)
      (for/list ([line (in-lines file)])
        (map string->number (string-split line))))))

(define dataset (file-lines->list "circled.txt"))

;-------------------------PREPROCESS------------------------------------

(define (change-class dataset prev after)
  (map (λ(x) (if (= prev (car x)) (cons after (cdr x)) x)) dataset))

(set! dataset (change-class (change-class dataset 3 1) 4 2))

(define (get-max lst)
  (apply max (map first lst)))

(define nr-zones (+ 1 (get-max dataset)))
(define dimension 2)

(define (codify target)
  (for/list ([i (in-range nr-zones)])
    (if (= i target) 1 0)))

(define (build lst)
  (for/list ([i (in-list lst)])
    (cons (codify (first i)) (map (λ(x) (exact->inexact (/ x 300))) (rest i)))))

(define items (build dataset))

;-------------------------BUILD NETWORK--------------------------------

(define (none x) x)
(define (sigmoid x) (/ 1 (+ 1 (exp (- x)))))
(define (d->sigmoid x) (* (sigmoid x) (- 1 (sigmoid x))))
(define SIGMOID (list sigmoid d->sigmoid))

(define (relu x) (if (< x 0) (* 0.01 x) x))
(define (d->relu x) (if (< x 0) 0.01 1))
(define RELU (list relu d->relu))

(define (selu x) (if (< x 0) (* 1.758094282 (sub1 (exp x))) (* 1.0507 x)))
(define (d->selu x) (if (< x 0) (* 1.758094282 (exp x)) 1.0507))
(define SELU (list selu d->selu))

(define (silu x) (/ x (+ 1 (exp (- x)))))
(define (d->silu x) (/ (add1 (* (add1 x) (exp (- x)))) (sqr (+ 1 (exp (- x))))))
(define SILU (list silu d->silu))

(define (gelu x [a 1.702]) (silu (* a x)))
(define (d->gelu x) (+ (sigmoid (* a x)) (* a x (d->sigmoid (* a x)))))
(define GELU (list gelu d->gelu))

(define (softplus x) (log (+ 1 (exp x))))
(define (d->softplus x) (/ 1 (+ 1 (exp (- x)))))
(define SOFTPLUS (list softplus d->softplus))

(define (dropr x) (drop-right x 1))
(define (transpose matrix) (apply map list matrix))
(define (dot x y) (apply + (map * x y)))

(define (build-network . structure)
  (let ([shape (map first structure)] [activation-functions (map second structure)])
    (for/list ([i (dropr shape)] [j (rest shape)] [f (rest activation-functions)])
      (list (build-list j (λ _ (build-list i (λ _ (/ (- (* 2 (random)) 1) (sqrt i))))))
            (build-list j (λ _ (/ (- (* 2 (random)) 1) (sqrt i))))
            f))))

(define (forward input layer)
  (let ([output (map + (map (λ(x) (apply + (map * x input))) (first layer)) (second layer))])
    (list (map (first (third layer)) output) (map (second (third layer)) output))))

(define (feed-forward input network)
  (let loop ([input input] [network network] [outputs empty])
    (cond
      [(empty? network) outputs]
      [else
       (define result (forward input (first network)))
       (loop (first result) (rest network) (cons result outputs))])))

(define (MSE output target)
  (* 0.5 (apply + (map (λ(x y) (sqr(- x y))) (first output) target))))

(define (d/output-biases output target)
  (map * (map - (first output) target) (second output)))

(define (d/cross-entropy output target)
  (map - (first output) target))

(define (d/hidden-biases reversed-weights:-1 -1:reversed-results d/output)
  (if (empty? reversed-weights:-1)
      (list d/output)
      (cons d/output
            (d/hidden-biases (rest reversed-weights:-1) (rest -1:reversed-results)
                             (map * 
                                  (map (λ(x) (dot x d/output)) (transpose (first reversed-weights:-1)))
                                  (first -1:reversed-results))))))

(define (get-derivatives deltas reversed-results:-1)
  (if (empty? deltas)
      empty
      (cons (list (map (λ(y) (map (λ(x) (* x y)) (first reversed-results:-1))) (first deltas))
                  (first deltas))
            (get-derivatives (rest deltas) (rest reversed-results:-1)))))

(define (update x y [step a])
  (map (λ(x y) (- x (* step y))) x y))
(define (limited-update x y [step a])
  (map (λ(x y) (- (* 0.999999 x) (* step y))) x y))

(define (update-layer layer layer-derivatives)
  (list (map update (first layer) (first layer-derivatives))
        (update (second layer) (second layer-derivatives))
        (third layer)))

(define (update-network network derivatives)
  (map update-layer network derivatives))

(define (go-epoch inputs network [Error 0])
  (cond
    [(empty? inputs) (list Error network)]
    [else
     (define input (first inputs))
     (define x (rest input))
     (define y (first input))
     (define results (feed-forward x network))
     (define output (first results))
     (define error (MSE output y))
     (define PragN (d/cross-entropy output y))
     (define Pragns (d/hidden-biases
                     (dropr (reverse (map first network))) (rest (map second results)) PragN))
     (define dxs (get-derivatives Pragns (append (rest (map first results)) (list x))))
     (go-epoch (rest inputs) (update-network network (reverse dxs)) (+ Error error))]))

(define a 0.009)
(define network (build-network (list dimension none)
                               (list 5 SELU) (list 5 SELU)
                               (list 5 SELU) (list nr-zones SIGMOID)))

;;---------------------------TEST NETWORK-------------------------------------

(define (test input network)
  (caar (feed-forward input network)))

(define (index-max lst)
  (index-of lst (argmax (λ(x) x) lst)))

(define (test-epoch trained-network test-set)
  (for/fold ([sum 0])
            ([i test-set])
    (if (= (index-max (test (rest i) trained-network)) (index-max (first i)))
        (+ sum 1)
        (+ sum 0))))

(define (train inputs network tests [n 0] [Error 0])
  (when (= (remainder n 25) 0)
    (printf "Epoch: ~a. Accuracy = ~a\n" n (exact->inexact (/ (test-epoch network tests) (length tests)))))
  (cond
    [(= n 200)
     (list Error network)]
    [else
     (define epoch-results (go-epoch inputs network))
     (train inputs (second epoch-results) tests (+ n 1) (first epoch-results))]))

(define trained (time (train (take items 10000) network (drop items 9000))))