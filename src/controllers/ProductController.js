export class ProductController {
    #productView;
    #currentUser = null;
    #events;
    #productService;
    constructor({
        productView,
        events,
        productService
    }) {
        this.#productView = productView;
        this.#productService = productService;
        this.#events = events;
        this.init();
    }

    static init(deps) {
        return new ProductController(deps);
    }

    async init() {
        this.setupCallbacks();
        this.setupEventListeners();
        const products = await this.#productService.getProducts();
        this.#productView.render(products, true);
    }

    setupEventListeners() {

        this.#events.onUserSelected((user) => {
            console.log("Product Controller dispatchRecommend")
            console.log(user);
            this.#currentUser = user;
            this.#productView.onUserSelected(user);
            this.#events.dispatchRecommend(user)
        });

        this.#events.onRecommendationsReady(({ recommendations }) => {
            this.#productView.render(recommendations, false);
        });
    }

    setupCallbacks() {
        this.#productView.registerBuyProductCallback(this.handleBuyProduct.bind(this));
    }

    async handleBuyProduct(product) {
        const user = this.#currentUser;
        this.#events.dispatchPurchaseAdded({ user, product });
    }

}
