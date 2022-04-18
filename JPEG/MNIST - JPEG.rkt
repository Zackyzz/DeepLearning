#lang racket

(define (file-lines->list path)
  (call-with-input-file path
    (λ (file)
      (for/list ([line (in-lines file)])
        (map string->number (string-split line ","))))))

(define dataset (file-lines->list "q100c6.txt"))

(define (get-max lst)
  (apply max (map first lst)))

(define nr-zones (+ 1 (get-max dataset)))
(define dimension (sub1 (length (first dataset))))

(define (codify target)
  (for/list ([i (in-range nr-zones)])
    (if (= i target) 1 0)))

(define (build lst)
  (for/list ([i (in-list lst)])
    (cons (codify (first i)) (map (λ(x) (/ x 255.0)) (rest i)))))

(define train-set (build (take dataset 7000)))
(define test-set (build (drop dataset 7000)))

;-------------------------BUILD NETWORK--------------------------------

(define (none x) x)
(define (sigmoid x) (/ 1 (+ 1 (exp (- x)))))
(define (d->sigmoid x) (* (sigmoid x) (- 1 (sigmoid x))))
(define SIGMOID (list sigmoid d->sigmoid))

(define (relu x) (if (< x 0) (* 0.01 x) x))
(define (d->relu x) (if (< x 0) 0.01 1))
(define RELU (list relu d->relu))

(define (selu x) (if (< x 0) (* 1.758 (sub1 (exp x))) (* 1.05 x)))
(define (d->selu x) (if (< x 0) (* 1.758 (exp x)) 1.05))
(define SELU (list selu d->selu))

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
  (* 0.5 (apply + (map (λ(x y) (sqr (- x y))) (first output) target))))

(define (d/MSE output target)
  (map * (map - (first output) target) (second output)))

(define (d/cross-entropy output target)
  (map - (first output) target))

(define (get-deltas weights outputs delta)
  (if (empty? weights)
      (list delta)
      (cons delta
            (get-deltas (rest weights) (rest outputs)
                        (map * (first outputs)
                             (map (λ(x) (dot x delta)) (transpose (first weights))))))))

(define (get-derivatives deltas outputs)
  (for/list ([i deltas] [j outputs])
    (list (map (λ(y) (map (λ(x) (* x y)) j)) i) i)))

(define (update x y [step a])
  (map (λ(x y) (- x (* step y))) x y))

(define (limited-update x y [step a])
  (map (λ(x y) (- (* 0.999999 x) (* step y))) x y))

(define (update-layer layer layer-derivatives)
  (list (map limited-update (first layer) (first layer-derivatives))
        (update (second layer) (second layer-derivatives))
        (third layer)))

(define (update-network network derivatives)
  (map update-layer network derivatives))

(define (go-epoch inputs network [Error 0])
  (cond
    [(empty? inputs) (list Error network)]
    [else
     (define input (first inputs))
     (define results (feed-forward (rest input) network))
     (define output (first results))
     (define deltas (get-deltas 
                     (dropr (reverse (map first network))) (rest (map second results))
                     (d/cross-entropy output (first input))))
     (define dxs (get-derivatives deltas (append (rest (map first results)) (list (rest input)))))
     (go-epoch (rest inputs) (update-network network (reverse dxs))
               (+ Error (MSE output (first input))))]))

(define a 0.01)
(define network (build-network (list dimension none) (list 90 RELU) (list 90 RELU)
                               (list 90 RELU) (list 90 RELU)
                               (list nr-zones SIGMOID)))

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
  (printf "Epoch: ~a. Accuracy = ~a\n" n (exact->inexact (/ (test-epoch network tests) (length tests))))
  (cond
    [(= n 100)
     (list Error network)]
    [else
     (define epoch-results (time (go-epoch inputs network)))
     (train inputs (second epoch-results) tests (+ n 1) (first epoch-results))]))

(define trained (train train-set network test-set))